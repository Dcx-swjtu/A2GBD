import torch
import torch.nn.functional as F
from torch_geometric.utils import degree, to_networkx
import numpy as np
try:
    import networkx as nx
except ImportError:
    nx = None
from typing import Tuple, List, Optional

def node_degree_stats(edge_index, num_nodes):
    """计算节点度统计"""
    # 确保在正确的设备上计算
    device = edge_index.device
    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float32)
    return deg.to(device)

def local_density_proxy(edge_index, num_nodes):
    """局部密度代理指标：度/平均度作为粗略估计"""
    deg = node_degree_stats(edge_index, num_nodes)
    mean_deg = deg.mean().clamp_min(1e-6)
    return (deg / mean_deg).clamp(max=10.0)

def spectral_features(data, k_eigen=10):
    """提取图的谱特征用于结构异常检测 - 优化GPU版本"""
    device = data.x.device if hasattr(data, 'x') else 'cpu'
    
    # 如果数据在GPU上，使用简化的谱特征计算避免CPU转换
    if device.type == 'cuda':
        # 使用度统计作为谱特征的近似
        degrees = node_degree_stats(data.edge_index, data.num_nodes)
        # 归一化度作为简化的谱特征
        normalized_degrees = (degrees - degrees.mean()) / (degrees.std() + 1e-6)
        # 填充到固定长度
        if len(normalized_degrees) < k_eigen:
            padding = torch.zeros(k_eigen - len(normalized_degrees), device=device)
            eigenvals = torch.cat([normalized_degrees, padding])
        else:
            eigenvals = normalized_degrees[:k_eigen]
        return eigenvals.to(device)
    else:
        # CPU版本保持原有逻辑
        G = to_networkx(data, to_undirected=True)
        try:
            # 拉普拉斯特征值
            L = nx.normalized_laplacian_matrix(G).astype(float)
            eigenvals = np.linalg.eigvals(L.toarray())
            eigenvals = np.sort(eigenvals)[:k_eigen]
            
            # 填充到固定长度
            if len(eigenvals) < k_eigen:
                eigenvals = np.pad(eigenvals, (0, k_eigen - len(eigenvals)), 'constant')
                
            return torch.tensor(eigenvals, dtype=torch.float32, device=device)
        except Exception as e:
            print(f"Warning: Error in spectral features computation: {e}")
            return torch.zeros(k_eigen, dtype=torch.float32, device=device)

def build_soft_edge_weight(edge_index, num_edges=None, device='cpu'):
    """构建可微分的软边权重"""
    if num_edges is None:
        num_edges = edge_index.size(1)
    # 确保设备一致性
    if hasattr(edge_index, 'device'):
        device = edge_index.device
    weights = torch.ones(num_edges, dtype=torch.float32, device=device)
    return weights

def apply_isolate_node(edge_index, edge_weight, node_id):
    """隔离指定节点（将其相关边权置零）"""
    device = edge_weight.device
    mask_keep = ~((edge_index[0] == node_id) | (edge_index[1] == node_id))
    edge_weight = edge_weight * mask_keep.float().to(device)
    return edge_weight

def weaken_edges(edge_weight, edge_indices, factor=0.5):
    """弱化指定边"""
    edge_weight[edge_indices] = edge_weight[edge_indices] * factor
    return edge_weight

def zero_edges(edge_weight, edge_indices):
    """删除指定边（置零）"""
    edge_weight[edge_indices] = 0.0
    return edge_weight

def node_incident_edges(edge_index, node_id):
    """找到与指定节点相关的所有边"""
    mask = (edge_index[0] == node_id) | (edge_index[1] == node_id)
    edge_indices = torch.nonzero(mask, as_tuple=False).view(-1)
    return edge_indices

def compute_graph_stats(data):
    """计算图的基本统计信息"""
    device = data.x.device if hasattr(data, 'x') else 'cpu'
    stats = {
        'num_nodes': data.num_nodes,
        'num_edges': data.edge_index.size(1),
        'avg_degree': float(data.edge_index.size(1) * 2 / data.num_nodes),
        'num_features': data.x.size(1) if hasattr(data, 'x') else 0,
        'num_classes': int(data.y.max().item() + 1) if hasattr(data, 'y') else 0
    }
    
    # 度分布 - 确保在正确设备上计算
    degrees = node_degree_stats(data.edge_index, data.num_nodes)
    stats.update({
        'degree_std': float(degrees.std()),
        'max_degree': float(degrees.max()),
        'min_degree': float(degrees.min())
    })
    
    return stats

def edge_similarity(data, edge_index, method='cosine'):
    """计算边的相似度用于重连策略"""
    if method == 'cosine':
        x_norm = F.normalize(data.x, dim=-1)
        src_feat = x_norm[edge_index[0]]
        dst_feat = x_norm[edge_index[1]]
        similarity = (src_feat * dst_feat).sum(dim=-1)
    elif method == 'euclidean':
        src_feat = data.x[edge_index[0]]
        dst_feat = data.x[edge_index[1]]
        similarity = -torch.norm(src_feat - dst_feat, dim=-1)
    else:
        raise ValueError(f"Unknown similarity method: {method}")
    
    return similarity

def subgraph_k_hop(data, center_nodes, k=2):
    """提取k跳子图"""
    from torch_geometric.utils import k_hop_subgraph
    
    if not isinstance(center_nodes, (list, tuple)):
        center_nodes = [center_nodes]
    
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        center_nodes, k, data.edge_index, relabel_nodes=True
    )
    
    return {
        'subset': subset,
        'edge_index': edge_index,
        'mapping': mapping,
        'edge_mask': edge_mask
    }

def motif_counting(data, motif_size=3):
    """简单的motif计数（三角形等）- 优化GPU版本"""
    device = data.x.device if hasattr(data, 'x') else 'cpu'
    
    if device.type == 'cuda':
        # GPU版本：使用简化的三角形计数近似
        # 基于度统计的三角形数量估计
        degrees = node_degree_stats(data.edge_index, data.num_nodes)
        # 简化的三角形数量估计：基于度的平方和
        triangle_estimate = (degrees * (degrees - 1) / 2).sum().item()
        return int(triangle_estimate / 3)  # 每个三角形被计算了3次
    else:
        # CPU版本保持原有逻辑
        G = to_networkx(data, to_undirected=True)
        
        if motif_size == 3:
            # 计算三角形数量
            triangles = list(nx.enumerate_all_cliques(G))
            triangle_count = len([t for t in triangles if len(t) == 3])
            return triangle_count
        else:
            # 更复杂的motif可以用专门的库
            return 0
