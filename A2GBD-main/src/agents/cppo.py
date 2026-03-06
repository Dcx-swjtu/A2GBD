import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

@dataclass
class PPOConfig:
    """PPO配置参数"""
    gamma: float = 0.99              # 折扣因子
    lam: float = 0.95                # GAE lambda
    clip_eps: float = 0.2            # PPO裁剪参数
    entropy_coef: float = 0.01       # 熵正则化系数
    value_coef: float = 0.5          # 价值损失系数
    lr: float = 3e-4                 # 学习率
    train_epochs: int = 4            # 每次更新的训练轮数
    minibatch_size: int = 64         # 小批次大小
    max_grad_norm: float = 0.5       # 梯度裁剪
    
    # CPPO约束相关
    cost_coef: float = 1.0           # 成本系数
    dual_lr: float = 1e-2            # 对偶变量学习率
    budget_c: float = 10.0           # 预算约束
    constraint_threshold: float = 0.1 # 约束违反阈值

class ActorCritic(nn.Module):
    """Actor-Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 安全Critic（用于约束）
        self.safety_critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
    def forward(self, state):
        shared_features = self.shared(state)
        
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features).squeeze(-1)
        safety_value = self.safety_critic(shared_features).squeeze(-1)
        
        # 确保value和safety_value的形状正确
        if value.dim() == 0:
            value = value.unsqueeze(0)
        if safety_value.dim() == 0:
            safety_value = safety_value.unsqueeze(0)
        
        # 数值稳定性检查
        if torch.isnan(action_logits).any():
            print("Warning: NaN detected in action_logits")
            action_logits = torch.zeros_like(action_logits)
        
        if torch.isnan(value).any():
            print("Warning: NaN detected in value")
            value = torch.zeros_like(value)
            
        if torch.isnan(safety_value).any():
            print("Warning: NaN detected in safety_value")
            safety_value = torch.zeros_like(safety_value)
        
        return action_logits, value, safety_value
    
    def get_action_and_value(self, state, action=None):
        action_logits, value, safety_value = self.forward(state)
        
        # 创建动作分布
        action_dist = D.Categorical(logits=action_logits)
        
        if action is None:
            action = action_dist.sample()
        
        action_logprob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        
        return action, action_logprob, value, safety_value, entropy

class CPPOAgent:
    """约束PPO智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig = None, device: str = 'cpu'):
        self.config = config or PPOConfig()
        self.device = device
        
        # 网络
        self.actor_critic = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.config.lr)
        
        # 对偶变量（用于约束）
        self.dual_lambda = torch.tensor(0.0, device=device, requires_grad=False)
        
        # 记录
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [], 
            'entropy_loss': [],
            'total_loss': [],
            'lambda_values': [],
            'constraint_violations': []
        }
    
    def select_action(self, state, deterministic=False):
        """选择动作 - 完全GPU版本，避免CPU-GPU数据传输"""
        self.actor_critic.eval()
        with torch.no_grad():
            # 确保状态是GPU张量
            if isinstance(state, torch.Tensor):
                if state.dim() == 1:
                    state = state.unsqueeze(0).to(self.device)
                else:
                    state = state.to(self.device)
            else:
                # 直接创建GPU张量，避免CPU中转
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            action_logits, value, safety_value = self.actor_critic(state)
            
            if deterministic:
                action = action_logits.argmax(dim=-1)
                action_logprob = torch.log_softmax(action_logits, dim=-1)[0, action]
            else:
                action_dist = D.Categorical(logits=action_logits)
                action = action_dist.sample()
                action_logprob = action_dist.log_prob(action)
            
            # 返回GPU张量，避免.item()调用
            return action, action_logprob, value, safety_value
    
    def compute_gae(self, rewards, values, dones, next_value=0.0):
        """计算广义优势估计(GAE) - CPU版本（保持兼容性）"""
        advantages = []
        gae = 0
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_val = values[step + 1]
            
            delta = rewards[step] + self.config.gamma * next_val * next_non_terminal - values[step]
            gae = delta + self.config.gamma * self.config.lam * next_non_terminal * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return advantages, returns
    
    def compute_gae_gpu(self, rewards, values, dones, next_value=0.0):
        """计算广义优势估计(GAE) - GPU优化版本"""
        # 确保所有输入都是GPU张量
        rewards = rewards.to(self.device)
        values = values.to(self.device)
        dones = dones.to(self.device)
        
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = torch.tensor(0.0, device=self.device)
        
        # 反向计算GAE
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step].float()
                next_val = torch.tensor(next_value, device=self.device)
            else:
                next_non_terminal = 1.0 - dones[step].float()
                next_val = values[step + 1]
            
            delta = rewards[step] + self.config.gamma * next_val * next_non_terminal - values[step]
            gae = delta + self.config.gamma * self.config.lam * next_non_terminal * gae
            advantages[step] = gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, trajectories: List[Dict], avg_cost: float):
        """更新智能体"""
        # 确保没有残留的计算图
        torch.autograd.set_detect_anomaly(False)
        
        # 强制清理任何残留的计算图
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 准备数据
        states = []
        actions = []
        old_logprobs = []
        rewards = []
        costs = []
        values = []
        safety_values = []
        dones = []
        
        for traj in trajectories:
            states.extend(traj['states'])
            actions.extend(traj['actions'])
            old_logprobs.extend(traj['logprobs'])
            rewards.extend(traj['rewards'])
            costs.extend(traj['costs'])
            values.extend(traj['values'])
            safety_values.extend(traj['safety_values'])
            dones.extend(traj['dones'])
        
        # 批量转换为GPU张量 - 优化版本，减少循环
        # 使用列表推导式直接创建GPU张量
        states = torch.stack([
            s.to(self.device) if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32, device=self.device)
            for s in states
        ])
        
        actions = torch.stack([
            a.to(self.device) if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=torch.long, device=self.device)
            for a in actions
        ])
        
        old_logprobs = torch.stack([
            lp.to(self.device) if isinstance(lp, torch.Tensor) else torch.tensor(lp, dtype=torch.float32, device=self.device)
            for lp in old_logprobs
        ])
        
        rewards = torch.stack([
            r.to(self.device) if isinstance(r, torch.Tensor) else torch.tensor(r, dtype=torch.float32, device=self.device)
            for r in rewards
        ])
        
        costs = torch.stack([
            c.to(self.device) if isinstance(c, torch.Tensor) else torch.tensor(c, dtype=torch.float32, device=self.device)
            for c in costs
        ])
        
        values = torch.stack([
            v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32, device=self.device)
            for v in values
        ])
        
        safety_values = torch.stack([
            sv.to(self.device) if isinstance(sv, torch.Tensor) else torch.tensor(sv, dtype=torch.float32, device=self.device)
            for sv in safety_values
        ])
        
        dones = torch.stack([
            d.to(self.device) if isinstance(d, torch.Tensor) else torch.tensor(d, dtype=torch.float32, device=self.device)
            for d in dones
        ])
        
        # 计算优势和回报 - GPU优化版本
        with torch.no_grad():
            # 主要奖励的优势 - 保持在GPU上
            advantages, returns = self.compute_gae_gpu(rewards, values, dones)
            
            # 成本的优势（用于约束）- 保持在GPU上
            cost_advantages, cost_returns = self.compute_gae_gpu(costs, safety_values, dones)
        
        # 标准化优势 - 安全处理小样本情况
        if advantages.numel() > 1:
            if advantages.std(unbiased=False) > 1e-8:
                advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False)
            else:
                advantages = advantages - advantages.mean()
        else:
            # 如果只有一个元素，直接设为0
            if advantages.numel() == 1:
                print(f"Warning: Only 1 advantage sample, setting to 0. Value: {advantages.item():.4f}")
            else:
                print(f"Warning: {advantages.numel()} advantage samples, setting to 0")
            advantages = torch.zeros_like(advantages)
            
        if cost_advantages.numel() > 1:
            if cost_advantages.std(unbiased=False) > 1e-8:
                cost_advantages = (cost_advantages - cost_advantages.mean()) / cost_advantages.std(unbiased=False)
            else:
                cost_advantages = cost_advantages - cost_advantages.mean()
        else:
            # 如果只有一个元素，直接设为0
            if cost_advantages.numel() == 1:
                print(f"Warning: Only 1 cost_advantage sample, setting to 0. Value: {cost_advantages.item():.4f}")
            else:
                print(f"Warning: {cost_advantages.numel()} cost_advantage samples, setting to 0")
            cost_advantages = torch.zeros_like(cost_advantages)
        
        # 如果样本太少，跳过训练
        if len(states) < 2:
            print("Warning: Too few samples, skipping training")
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'total_loss': 0.0,
                'lambda': self.dual_lambda.item(),
                'constraint_violation': avg_cost - self.config.budget_c
            }
        
        # 更新网络
        for epoch in range(self.config.train_epochs):
            # 创建小批次
            indices = torch.randperm(len(states))
            
            # 确保每次epoch都使用新的计算图
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 确保没有残留的计算图
            torch.autograd.set_detect_anomaly(False)
            
            # 强制清理任何残留的计算图
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 重置网络的计算图状态
            for param in self.actor_critic.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()
            
            for start in range(0, len(states), self.config.minibatch_size):
                end = start + self.config.minibatch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices].detach().clone()
                batch_actions = actions[batch_indices].detach().clone()
                batch_old_logprobs = old_logprobs[batch_indices].detach().clone()
                batch_advantages = advantages[batch_indices].detach().clone()
                batch_cost_advantages = cost_advantages[batch_indices].detach().clone()
                batch_returns = returns[batch_indices].detach().clone()
                batch_cost_returns = cost_returns[batch_indices].detach().clone()
                
                # 确保每次迭代都使用新的计算图
                torch.autograd.set_detect_anomaly(False)
                
                # 前向传播 - 确保每次都是新的计算图
                with torch.enable_grad():
                    _, new_logprobs, new_values, new_safety_values, entropy = \
                        self.actor_critic.get_action_and_value(batch_states, batch_actions)
                
                # PPO损失
                ratio = (new_logprobs - batch_old_logprobs).exp()
                
                # 策略损失（主要目标）
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 约束项（CPPO）
                constraint_surr1 = ratio * batch_cost_advantages
                constraint_surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * batch_cost_advantages
                constraint_loss = torch.max(constraint_surr1, constraint_surr2).mean()
                
                # 价值损失 - 确保形状匹配
                # 强制确保形状匹配
                if new_values.dim() > 1:
                    new_values = new_values.squeeze()
                if new_safety_values.dim() > 1:
                    new_safety_values = new_safety_values.squeeze()
                
                # 确保targets也是正确的形状
                if batch_returns.dim() > 1:
                    batch_returns = batch_returns.squeeze()
                if batch_cost_returns.dim() > 1:
                    batch_cost_returns = batch_cost_returns.squeeze()
                
                # 最终形状检查和修复
                if new_values.shape != batch_returns.shape:
                    # 强制调整形状
                    if new_values.numel() == batch_returns.numel():
                        new_values = new_values.view(batch_returns.shape)
                    else:
                        # 如果元素数量不匹配，使用均值
                        value_loss = F.mse_loss(new_values.mean(), batch_returns.mean())
                        safety_value_loss = F.mse_loss(new_safety_values.mean(), batch_cost_returns.mean())
                        # 跳过这次训练
                        continue
                
                if new_safety_values.shape != batch_cost_returns.shape:
                    # 强制调整形状
                    if new_safety_values.numel() == batch_cost_returns.numel():
                        new_safety_values = new_safety_values.view(batch_cost_returns.shape)
                    else:
                        # 如果元素数量不匹配，使用均值
                        value_loss = F.mse_loss(new_values.mean(), batch_returns.mean())
                        safety_value_loss = F.mse_loss(new_safety_values.mean(), batch_cost_returns.mean())
                        # 跳过这次训练
                        continue
                
                # 计算损失
                try:
                    value_loss = F.mse_loss(new_values, batch_returns)
                    safety_value_loss = F.mse_loss(new_safety_values, batch_cost_returns)
                except RuntimeError as e:
                    print(f"Error in loss calculation: {e}")
                    print(f"Shapes: new_values={new_values.shape}, batch_returns={batch_returns.shape}")
                    print(f"Shapes: new_safety_values={new_safety_values.shape}, batch_cost_returns={batch_cost_returns.shape}")
                    # 使用均值作为fallback
                    value_loss = F.mse_loss(new_values.mean(), batch_returns.mean())
                    safety_value_loss = F.mse_loss(new_safety_values.mean(), batch_cost_returns.mean())
                
                # 熵损失
                entropy_loss = -entropy.mean()
                
                # 总损失 - 确保dual_lambda不参与梯度计算
                dual_lambda_detached = self.dual_lambda.detach()
                total_loss = (policy_loss + 
                             self.config.value_coef * (value_loss + safety_value_loss) +
                             self.config.entropy_coef * entropy_loss +
                             dual_lambda_detached * constraint_loss)
                
                # 反向传播 - 确保每次都是新的计算图
                self.optimizer.zero_grad()
                total_loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # 强制清理计算图
                total_loss.detach_()
                
                # 记录统计信息 - 批量转换为CPU，减少数据传输
                with torch.no_grad():
                    self.training_stats['policy_loss'].append(policy_loss.item())
                    self.training_stats['value_loss'].append((value_loss + safety_value_loss).item())
                    self.training_stats['entropy_loss'].append(entropy_loss.item())
                    self.training_stats['total_loss'].append(total_loss.item())
                
                # 清理计算图，防止内存泄漏
                del total_loss, policy_loss, value_loss, safety_value_loss, entropy_loss, constraint_loss
                del ratio, surr1, surr2, constraint_surr1, constraint_surr2
                del new_logprobs, new_values, new_safety_values, entropy
                
                # 强制清理计算图
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
                # 确保没有残留的计算图
                torch.autograd.set_detect_anomaly(False)
        
        # 更新对偶变量 - 确保不参与梯度计算
        constraint_violation = avg_cost - self.config.budget_c
        with torch.no_grad():
            self.dual_lambda = torch.clamp(
                self.dual_lambda + self.config.dual_lr * constraint_violation,
                min=0.0
            )
        
        # 批量记录统计信息，减少CPU-GPU传输
        with torch.no_grad():
            self.training_stats['lambda_values'].append(self.dual_lambda.item())
            self.training_stats['constraint_violations'].append(constraint_violation)
        
        # 返回统计信息 - 一次性计算所有统计量
        with torch.no_grad():
            recent_policy_losses = self.training_stats['policy_loss'][-10:]
            recent_value_losses = self.training_stats['value_loss'][-10:]
            recent_entropy_losses = self.training_stats['entropy_loss'][-10:]
            recent_total_losses = self.training_stats['total_loss'][-10:]
            
            return {
                'policy_loss': np.mean(recent_policy_losses) if recent_policy_losses else 0.0,
                'value_loss': np.mean(recent_value_losses) if recent_value_losses else 0.0,
                'entropy': -np.mean(recent_entropy_losses) if recent_entropy_losses else 0.0,
                'total_loss': np.mean(recent_total_losses) if recent_total_losses else 0.0,
                'lambda': self.dual_lambda.item(),
                'constraint_violation': constraint_violation
            }
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dual_lambda': self.dual_lambda,
            'config': self.config,
            'training_stats': self.training_stats
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.dual_lambda = checkpoint['dual_lambda'].to(self.device)
        self.training_stats = checkpoint.get('training_stats', {})
        
        return checkpoint.get('config', self.config)
    
    def clear_gpu_cache(self):
        """清理GPU缓存以释放内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
