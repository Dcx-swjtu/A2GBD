import numpy as np
import torch
import logging
import os
from typing import Optional, Dict, Any

class EarlyStopping:
    """早停训练工具类
    
    参数:
        patience (int): 在停止训练前等待的epoch数量
        min_delta (float): 被视为改进的最小变化量
        mode (str): 'min' 表示监控指标越小越好，'max' 表示监控指标越大越好
        verbose (bool): 是否打印早停信息
        save_path (str): 模型保存路径
    """
    
    def __init__(self, 
                 patience: int = 10, 
                 min_delta: float = 0.0, 
                 mode: str = 'max',
                 verbose: bool = True,
                 save_path: Optional[str] = None):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None
        
        if mode not in ['min', 'max']:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
        
        # 根据模式设置比较函数
        self.improved = self._improved_max if mode == 'max' else self._improved_min
    
    def _improved_max(self, score: float) -> bool:
        """检查在max模式下是否有改进"""
        return self.best_score is None or score > self.best_score + self.min_delta
    
    def _improved_min(self, score: float) -> bool:
        """检查在min模式下是否有改进"""
        return self.best_score is None or score < self.best_score - self.min_delta
    
    def __call__(self, score: float, model: torch.nn.Module = None, extra_info: Dict[str, Any] = None):
        """检查是否应该早停
        
        参数:
            score (float): 当前评估指标
            model (torch.nn.Module, optional): 要保存的模型
            extra_info (dict, optional): 要与模型一起保存的额外信息
            
        返回:
            bool: 是否应该停止训练
        """
        if self.improved(score):
            # 有改进，重置计数器
            if self.verbose:
                if self.best_score is not None:
                    logging.info(f"Validation score improved ({self.best_score:.6f} --> {score:.6f})")
                else:
                    logging.info(f"Validation score: {score:.6f}")
            
            self.best_score = score
            self.counter = 0
            
            # 保存模型
            if model is not None and self.save_path is not None:
                self._save_checkpoint(model, extra_info)
        else:
            # 没有改进，增加计数器
            self.counter += 1
            if self.verbose:
                logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            
            # 检查是否应该停止
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logging.info("Early stopping triggered")
        
        return self.early_stop
    
    def _save_checkpoint(self, model: torch.nn.Module, extra_info: Optional[Dict[str, Any]] = None):
        """保存模型检查点"""
        if self.save_path is None:
            return
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        # 准备保存内容
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'best_score': self.best_score
        }
        
        # 添加额外信息
        if extra_info is not None:
            checkpoint.update(extra_info)
        
        # 保存模型
        torch.save(checkpoint, self.save_path)
        if self.verbose:
            logging.info(f"Model saved to {self.save_path}")
    
    def load_best_model(self, model: torch.nn.Module):
        """加载最佳模型"""
        if self.save_path is None or not os.path.exists(self.save_path):
            logging.warning("No saved model found")
            return None
        
        checkpoint = torch.load(self.save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
