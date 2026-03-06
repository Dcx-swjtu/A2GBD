import torch
import numpy as np
from typing import Dict, Any

class AdaptiveParameterManager:
    """根据图规模自适应调整超参数"""
    
    def __init__(self):
        self.size_thresholds = {
            'small': 5000,    # 节点数 < 5000
            'medium': 20000,  # 5000 <= 节点数 < 20000  
            'large': float('inf')  # 节点数 >= 20000
        }
    
    def get_adaptive_params(self, num_nodes: int, num_edges: int, 
                          num_features: int, num_classes: int) -> Dict[str, Any]:
        """根据图规模返回自适应参数"""
        
        # 确定图规模类别
        if num_nodes < self.size_thresholds['small']:
            size_category = 'small'
        elif num_nodes < self.size_thresholds['medium']:
            size_category = 'medium'
        else:
            size_category = 'large'
        
        # 基础参数
        base_params = {
            'small': {
                'topk': 32,
                'max_steps_per_episode': 32,
                'budget_ratio': 0.005,
                'lambda_asr': 1.5,
                'lambda_acc': 1.0,
                'lambda_cost': 0.1,
                'lr_rl': 3e-4,
                'dual_lr': 1e-2,
                'budget_c': 10.0,
                'ppo_epochs': 4,
                'minibatch_size': 64,
                'mc_samples': 12,
                'eval_frequency': 10
            },
            'medium': {
                'topk': 48,  # 增加候选节点数
                'max_steps_per_episode': 48,
                'budget_ratio': 0.003,  # 降低预算比例
                'lambda_asr': 2.0,  # 增强ASR惩罚
                'lambda_acc': 1.2,
                'lambda_cost': 0.15,
                'lr_rl': 2e-4,  # 降低学习率
                'dual_lr': 5e-3,
                'budget_c': 15.0,  # 增加预算
                'ppo_epochs': 6,  # 增加训练轮数
                'minibatch_size': 128,
                'mc_samples': 16,  # 增加MC采样
                'eval_frequency': 15
            },
            'large': {
                'topk': 96,  # 大幅增加候选节点
                'max_steps_per_episode': 48,  # 减少步数避免过长episode
                'budget_ratio': 0.001,  # 更严格的预算
                'lambda_asr': 3.0,  # 更强的ASR惩罚
                'lambda_acc': 2.0,  # 更强的准确率奖励
                'lambda_cost': 0.3,  # 更强的成本惩罚
                'lr_rl': 5e-5,  # 更小的学习率
                'dual_lr': 1e-3,
                'budget_c': 25.0,  # 增加预算
                'ppo_epochs': 10,  # 增加训练轮数
                'minibatch_size': 512,  # 增加批次大小
                'mc_samples': 24,  # 增加MC采样
                'eval_frequency': 25  # 减少评估频率
            }
        }
        
        params = base_params[size_category].copy()
        
        # 根据具体特征进一步调整
        params.update(self._adjust_by_features(num_nodes, num_edges, num_features, num_classes))
        
        return params
    
    def _adjust_by_features(self, num_nodes: int, num_edges: int, 
                          num_features: int, num_classes: int) -> Dict[str, Any]:
        """根据具体特征进一步调整参数"""
        adjustments = {}
        
        # 根据节点密度调整
        density = num_edges / num_nodes
        if density > 10:  # 高密度图
            adjustments['budget_ratio'] = 1.5  # 增加预算
            adjustments['lambda_cost'] = 0.8   # 降低成本惩罚
        
        # 根据特征维度调整
        if num_features > 2000:  # 高维特征
            adjustments['mc_samples'] = 16  # 增加MC采样
            adjustments['lambda_acc'] = 1.2  # 增强准确率奖励
        
        # 根据类别数调整
        if num_classes > 5:  # 多分类问题
            adjustments['lambda_asr'] = 1.3  # 增强ASR惩罚
            adjustments['topk'] = 48  # 增加候选节点数
        
        return adjustments

def get_adaptive_config(data_stats: Dict[str, Any]) -> Dict[str, Any]:
    """便捷函数：根据数据统计获取自适应配置"""
    manager = AdaptiveParameterManager()
    return manager.get_adaptive_params(
        num_nodes=data_stats['num_nodes'],
        num_edges=data_stats['num_edges'], 
        num_features=data_stats['num_features'],
        num_classes=data_stats['num_classes']
    )
