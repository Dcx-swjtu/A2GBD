import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Tuple, Dict, Optional, List
import numpy as np

from ..utils.graph_ops import local_density_proxy, spectral_features
from ..models.gnn import GCN, GAT

@torch.no_grad()
def compute_uncertainty_scores(model, data: Data, edge_weight: Optional[torch.Tensor] = None, 
                              mc_samples: int = 12) -> Tuple[torch.Tensor, Dict]:
    """
    计算基于MC-Dropout的不确定性分数
    
    Returns:
        uncertainty_scores: [N] 不确定性分数
        aux_info: 包含熵、BALD等辅助信息的字典
    """
    # MC-Dropout 预测熵和BALD
    entropy, bald, all_probs = model.mc_predict_entropy(data, edge_weight=edge_weight, mc_samples=mc_samples) # 使用传入的mc_samples
    
    # 组合不确定性分数 (增加BALD权重，提高对属性攻击的敏感度)
    uncertainty = 0.3 * entropy + 0.7 * bald.clamp_min(0.0)
    
    # 使用 all_probs 计算预测均值和标准差，避免重复前向传播
    predictive_mean = all_probs.mean(dim=0)
    predictive_std = all_probs.std(dim=0).mean(dim=-1)  # 平均跨类别的标准差
    
    aux_info = {
        'entropy': entropy,
        'bald': bald,
        'predictive_mean': predictive_mean,
        'predictive_std': predictive_std
    }
    
    return uncertainty.detach(), aux_info

@torch.no_grad() 
def compute_structural_scores(data: Data, method: str = 'density') -> torch.Tensor:
    """
    计算结构异常分数
    
    Args:
        data: 图数据
        method: 计算方法 ('density', 'spectral', 'degree_zscore')
    
    Returns:
        structural_scores: [N] 结构异常分数
    """
    # 确保在正确的设备上计算
    device = data.x.device
    
    if method == 'density':
        # 局部密度异常
        density = local_density_proxy(data.edge_index, data.num_nodes)
        density_mean = density.mean()
        density_std = density.std() + 1e-6
        scores = torch.abs(density - density_mean) / density_std
        
    elif method == 'degree_zscore':
        # 度的z-score
        from ..utils.graph_ops import node_degree_stats
        degrees = node_degree_stats(data.edge_index, data.num_nodes).float()
        degree_mean = degrees.mean()
        degree_std = degrees.std() + 1e-6
        scores = torch.abs(degrees - degree_mean) / degree_std
        
    elif method == 'spectral':
        # 基于谱特征的异常检测（简化版）
        try:
            # 提取每个节点的局部谱特征
            scores = torch.zeros(data.num_nodes, device=device)
            # 这里可以实现更复杂的谱异常检测
            # 简化：使用度作为proxy
            from ..utils.graph_ops import node_degree_stats
            degrees = node_degree_stats(data.edge_index, data.num_nodes).float()
            scores = (degrees - degrees.median()).abs()
        except Exception:
            # 回退到度统计
            from ..utils.graph_ops import node_degree_stats
            degrees = node_degree_stats(data.edge_index, data.num_nodes).float()
            scores = degrees
    else:
        raise ValueError(f"Unknown structural scoring method: {method}")
    
    return scores.clamp_min(0.0).to(device)

@torch.no_grad()
def compute_influence_scores(model, data: Data, target_nodes: Optional[List[int]] = None,
                           method: str = 'gradient') -> torch.Tensor:
    """
    计算影响分数（基于梯度或其他方法）
    
    Args:
        model: 训练好的模型
        data: 图数据
        target_nodes: 目标节点列表（如果为None则考虑所有节点）
        method: 影响计算方法
    
    Returns:
        influence_scores: [N] 影响分数
    """
    if target_nodes is None:
        target_nodes = list(range(data.num_nodes))
    
    model.eval()
    model.zero_grad()
    
    # 计算总损失对节点特征的梯度
    logits = model(data.x, data.edge_index)
    
    if hasattr(data, 'train_mask') and data.train_mask.sum() > 0:
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    else:
        # 如果没有train_mask，使用所有节点
        loss = F.cross_entropy(logits, data.y)
    
    # 计算梯度
    grad_x = torch.autograd.grad(loss, data.x, retain_graph=False, create_graph=False)[0]
    
    # 影响分数 = 梯度范数
    influence_scores = grad_x.norm(dim=-1)
    
    return influence_scores.detach()

def compute_al_scores(model, data: Data, edge_weight: Optional[torch.Tensor] = None,
                     weights: Dict[str, float] = None, mc_samples: int = 12) -> Tuple[torch.Tensor, Dict]:
    """
    计算主动学习综合分数
    
    Args:
        model: 训练好的模型
        data: 图数据
        edge_weight: 边权重
        weights: 各组件权重 {'uncertainty': 1.0, 'structural': 0.5, 'influence': 0.3}
        mc_samples: MC采样次数
    
    Returns:
        al_scores: [N] 主动学习分数
        aux_info: 辅助信息字典
    """
    if weights is None:
        # 调整权重：增加不确定性和结构权重，降低影响力权重
        weights = {'uncertainty': 1.2, 'structural': 0.8, 'influence': 0.2}
    
    # 确保在正确的设备上计算
    device = data.x.device
    
    # 1. 不确定性分数
    uncertainty_scores, uncertainty_aux = compute_uncertainty_scores(
        model, data, edge_weight, mc_samples
    )
    
    # 2. 结构异常分数
    structural_scores = compute_structural_scores(data, method='density')
    
    # 3. 影响分数 - 简化版本，避免梯度计算
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, edge_weight)
        probs = logits.softmax(dim=-1)
        max_probs, _ = probs.max(dim=-1)
        influence_scores = 1.0 - max_probs  # 低置信度 = 高影响
    
    # 确保所有分数都在同一设备上
    uncertainty_scores = uncertainty_scores.to(device)
    structural_scores = structural_scores.to(device)
    influence_scores = influence_scores.to(device)
    
    # 归一化各组件分数
    uncertainty_norm = (uncertainty_scores - uncertainty_scores.mean()) / (uncertainty_scores.std() + 1e-6)
    structural_norm = (structural_scores - structural_scores.mean()) / (structural_scores.std() + 1e-6)
    influence_norm = (influence_scores - influence_scores.mean()) / (influence_scores.std() + 1e-6)
    
    # 组合分数
    al_scores = (weights['uncertainty'] * uncertainty_norm.clamp(-3, 3) + 
                weights['structural'] * structural_norm.clamp(-3, 3) +
                weights['influence'] * influence_norm.clamp(-3, 3))
    
    # 确保分数非负
    al_scores = al_scores - al_scores.min() + 1e-6
    
    aux_info = {
        'uncertainty': uncertainty_scores,
        'structural': structural_scores,
        'influence': influence_scores,
        'uncertainty_aux': uncertainty_aux,
        'weights': weights
    }
    
    return al_scores, aux_info

@torch.no_grad()
def select_topk_candidates(model, data: Data, topk: int = 32, 
                          edge_weight: Optional[torch.Tensor] = None,
                          mask: Optional[torch.Tensor] = None,
                          selection_method: str = 'greedy',
                          **kwargs) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    选择Top-K候选节点用于后续处理
    
    Args:
        model: 训练好的模型
        data: 图数据
        topk: 选择的候选数量
        edge_weight: 边权重
        mask: 可选择节点的mask（True表示可选择）
        selection_method: 选择方法 ('greedy', 'diverse', 'uncertainty_only')
        **kwargs: 其他参数
    
    Returns:
        selected_indices: [topk] 选中的节点索引
        selected_scores: [topk] 对应的分数
        aux_info: 辅助信息
    """
    # 计算AL分数
    al_scores, aux_info = compute_al_scores(model, data, edge_weight, **kwargs)
    
    # 应用mask
    if mask is not None:
        al_scores = al_scores.masked_fill(~mask, float('-inf'))
    
    # 选择策略
    if selection_method == 'greedy':
        # 简单贪心：选择分数最高的topk个
        values, indices = torch.topk(al_scores, k=min(topk, al_scores.numel()))
        
    elif selection_method == 'diverse':
        # 多样性选择：考虑特征相似性，避免选择过于相似的节点
        values, candidates = torch.topk(al_scores, k=min(topk * 3, al_scores.numel()))
        
        # 简化的多样性选择：基于特征距离
        selected = [candidates[0].item()]  # 先选分数最高的
        selected_indices = [candidates[0]]
        selected_values = [values[0]]
        
        for i in range(1, min(topk, len(candidates))):
            max_min_dist = -1
            best_candidate = None
            best_value = None
            
            for j, candidate in enumerate(candidates[1:], 1):
                if candidate.item() in selected:
                    continue
                
                # 计算与已选节点的最小距离
                candidate_feat = data.x[candidate]
                min_dist = float('inf')
                
                for sel_idx in selected:
                    sel_feat = data.x[sel_idx]
                    dist = torch.norm(candidate_feat - sel_feat).item()
                    min_dist = min(min_dist, dist)
                
                # 选择最小距离最大的候选（多样性）
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_candidate = candidate
                    best_value = values[j]
            
            if best_candidate is not None:
                selected.append(best_candidate.item())
                selected_indices.append(best_candidate)
                selected_values.append(best_value)
        
        indices = torch.stack(selected_indices)
        values = torch.stack(selected_values)
        
    elif selection_method == 'uncertainty_only':
        # 仅基于不确定性选择
        uncertainty_scores = aux_info['uncertainty']
        if mask is not None:
            uncertainty_scores = uncertainty_scores.masked_fill(~mask, float('-inf'))
        values, indices = torch.topk(uncertainty_scores, k=min(topk, uncertainty_scores.numel()))
        
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    
    # 过滤掉无效的选择
    valid_mask = values != float('-inf')
    indices = indices[valid_mask]
    values = values[valid_mask]
    
    aux_info['selection_method'] = selection_method
    aux_info['total_candidates'] = al_scores.numel()
    aux_info['selected_count'] = len(indices)
    
    return indices, values, aux_info

def update_al_budget(current_budget: int, used_queries: int, 
                    budget_strategy: str = 'fixed') -> int:
    """
    更新主动学习预算
    
    Args:
        current_budget: 当前预算
        used_queries: 已使用的查询次数
        budget_strategy: 预算策略
    
    Returns:
        new_budget: 新的预算
    """
    if budget_strategy == 'fixed':
        return max(0, current_budget - used_queries)
    elif budget_strategy == 'adaptive':
        # 自适应预算：根据效果动态调整
        # 这里简化为固定策略
        return max(0, current_budget - used_queries)
    else:
        return current_budget
