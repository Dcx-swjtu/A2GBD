import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from typing import Optional

class GCN(nn.Module):
    """基础GCN模型，支持MC-Dropout用于主动学习不确定性估计"""
    
    def __init__(self, in_dim: int, hid: int = 64, out_dim: int = 7, 
                 dropout: float = 0.5, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hid, add_self_loops=True, normalize=True))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hid, hid, add_self_loops=True, normalize=True))
            
        self.convs.append(GCNConv(hid, out_dim, add_self_loops=True, normalize=True))
        
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        # 确保所有张量在同一设备上
        device = x.device
        edge_index = edge_index.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)
            
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        x = self.convs[-1](x, edge_index, edge_weight)
        return x

    @torch.no_grad()
    def predict(self, data, edge_weight=None):
        """标准预测（关闭dropout）"""
        self.eval()
        logits = self.forward(data.x, data.edge_index, edge_weight)
        return logits.softmax(dim=-1)

    def mc_predict_entropy(self, data, edge_weight=None, mc_samples=12):
        """MC-Dropout预测熵，用于主动学习不确定性估计"""
        self.train()  # 启用dropout
        probs = []
        
        try:
            with torch.no_grad():
                for _ in range(mc_samples):
                    logits = self.forward(data.x, data.edge_index, edge_weight)
                    probs.append(logits.softmax(dim=-1))
            
            probs = torch.stack(probs, dim=0)  # [mc_samples, N, C]
            prob_mean = probs.mean(dim=0)      # [N, C]
            
            # 预测熵 H[y|x]
            entropy = -(prob_mean * (prob_mean.clamp_min(1e-9)).log()).sum(dim=-1)
            
            # BALD近似: H[y|x] - E_θ[H[y|x,θ]]
            entropy_mc = -(probs * (probs.clamp_min(1e-9)).log()).sum(dim=-1).mean(dim=0)
            bald = entropy - entropy_mc
            
            return entropy.detach(), bald.detach(), probs.detach()
            
        except Exception as e:
            # 如果出现CUDA错误，返回零值
            print(f"Warning: Error in MC prediction: {e}")
            device = data.x.device
            num_nodes = data.x.size(0)
            num_classes = data.y.max().item() + 1 if hasattr(data, 'y') else 7
            
            zero_entropy = torch.zeros(num_nodes, device=device)
            zero_bald = torch.zeros(num_nodes, device=device)
            zero_probs = torch.zeros(mc_samples, num_nodes, num_classes, device=device)
            
            return zero_entropy, zero_bald, zero_probs


class GAT(nn.Module):
    """GAT模型用于对比实验"""
    
    def __init__(self, in_dim: int, hid: int = 64, out_dim: int = 7, 
                 dropout: float = 0.5, heads: int = 8):
        super().__init__()
        self.dropout = dropout
        
        self.conv1 = GATConv(in_dim, hid, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hid * heads, out_dim, heads=1, concat=False, dropout=dropout)
        
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        # 确保所有张量在同一设备上
        device = x.device
        edge_index = edge_index.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)
            
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout_layer(x)
        x = self.conv2(x, edge_index)
        return x

    @torch.no_grad()
    def predict(self, data, edge_weight=None):
        self.eval()
        logits = self.forward(data.x, data.edge_index, edge_weight)
        return logits.softmax(dim=-1)

    def mc_predict_entropy(self, data, edge_weight=None, mc_samples=12):
        self.train()
        probs = []
        
        try:
            with torch.no_grad():
                for _ in range(mc_samples):
                    logits = self.forward(data.x, data.edge_index, edge_weight)
                    probs.append(logits.softmax(dim=-1))
            
            probs = torch.stack(probs, dim=0)
            
            prob_mean = probs.mean(dim=0)
            entropy = -(prob_mean * (prob_mean.clamp_min(1e-9)).log()).sum(dim=-1)
            entropy_mc = -(probs * (probs.clamp_min(1e-9)).log()).sum(dim=-1).mean(dim=0)
            bald = entropy - entropy_mc
            
            return entropy.detach(), bald.detach(), probs.detach()
            
        except Exception as e:
            # 如果出现CUDA错误，返回零值
            print(f"Warning: Error in GAT MC prediction: {e}")
            device = data.x.device
            num_nodes = data.x.size(0)
            num_classes = data.y.max().item() + 1 if hasattr(data, 'y') else 7
            
            zero_entropy = torch.zeros(num_nodes, device=device)
            zero_bald = torch.zeros(num_nodes, device=device)
            zero_probs = torch.zeros(mc_samples, num_nodes, num_classes, device=device)
            
            return zero_entropy, zero_bald, zero_probs


class MINEEstimator(nn.Module):
    """MINE互信息估计器，用于信息论奖励"""
    
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, y):
        """
        x: [N, x_dim] 触发器特征
        y: [N, y_dim] 标签特征（one-hot或embedding）
        """
        xy = torch.cat([x, y], dim=-1)
        return self.net(xy).squeeze(-1)
    
    def mi_estimate(self, x, y):
        """估计I(X;Y)"""
        # 正样本：真实配对
        joint_score = self.forward(x, y)
        
        # 负样本：随机配对
        y_shuffle = y[torch.randperm(y.size(0))]
        marginal_score = self.forward(x, y_shuffle)
        
        # MINE目标
        mi = joint_score.mean() - torch.log(marginal_score.exp().mean() + 1e-8)
        return mi
