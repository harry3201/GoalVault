# backend/distillation/trainer.py
"""
Knowledge Distillation Training Module
Implements various distillation techniques for financial goal prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import logging
from tqdm import tqdm
import wandb
from pathlib import Path

from ..models.teacher_model import FinancialGoalTeacher
from ..models.student_model import FinancialGoalStudent

@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation"""
    temperature: float = 4.0
    alpha: float = 0.7  # Weight for distillation loss
    beta: float = 0.3   # Weight for task loss
    feature_weight: float = 0.1  # Weight for feature matching
    attention_weight: float = 0.05  # Weight for attention transfer
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    warmup_epochs: int = 10
    
    # Loss weights for different tasks
    task_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.task_weights is None:
            self.task_weights = {
                'spending': 1.0,
                'savings': 1.0,
                'goal_achievement': 2.0,  # Higher weight for main task
                'timeline': 1.0,
                'risk': 0.5,
                'recommendation': 0.5
            }

class AdvancedDistillationLoss(nn.Module):
    """Advanced distillation loss with multiple components"""
    
    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config
        self.temperature = config.temperature
        self.alpha = config.alpha
        self.beta = config.beta
        
    def forward(self, 
                student_outputs: Dict[str, torch.Tensor],
                teacher_outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                student_features: Dict[str, torch.Tensor],
                teacher_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive distillation loss
        """
        losses = {}
        
        # 1. Soft target distillation (KL divergence)
        distillation_losses = self._compute_soft_target_loss(
            student_outputs, teacher_outputs
        )
        losses.update(distillation_losses)
        
        # 2. Hard target loss (ground truth)
        task_losses = self._compute_task_loss(student_outputs, targets)
        losses.update(task_losses)
        
        # 3. Feature matching loss
        feature_loss = self._compute_feature_matching_loss(
            student_features, teacher_features
        )
        losses['feature_matching'] = feature_loss
        
        # 4. Attention transfer loss
        attention_loss = self._compute_attention_transfer_loss(
            student_outputs.get('attention_weights'),
            teacher_outputs.get('attention_weights')
        )
        losses['attention_transfer'] = attention_loss
        
        # 5. Representation similarity loss
        representation_loss = self._compute_representation_similarity_loss(
            student_outputs.get('main_representation'),
            teacher_outputs.get('main_representation')
        )
        losses['representation_similarity'] = representation_loss
        
        # Combine all losses
        total_loss = self._combine_losses(losses)
        losses['total'] = total_loss
        
        return losses
    
    def _compute_soft_target_loss(self, 
                                 student_outputs: Dict[str, torch.Tensor],
                                 teacher_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute soft target distillation losses"""
        losses = {}
        
        # Spending prediction (regression)
        if 'spending_prediction' in student_outputs:
            losses['distill_spending'] = F.mse_loss(
                student_outputs['spending_prediction'],
                teacher_outputs['spending_prediction'].detach()
            )
        
        # Savings prediction (regression)
        if 'savings_prediction' in student_outputs:
            losses['distill_savings'] = F.mse_loss(
                student_outputs['savings_prediction'],
                teacher_outputs['savings_prediction'].detach()
            )
        
        # Goal achievement (probability)
        if 'goal_achievement' in student_outputs:
            losses['distill_goal_achievement'] = F.mse_loss(
                student_outputs['goal_achievement'],
                teacher_outputs['goal_achievement'].detach()
            )
        
        # Timeline prediction (regression)
        if 'goal_timeline' in student_outputs:
            losses['distill_timeline'] = F.mse_loss(
                student_outputs['goal_timeline'],
                teacher_outputs['goal_timeline'].detach()
            )
        
        # Risk assessment (soft labels)
        if 'risk_assessment' in student_outputs:
            teacher_risk_soft = F.softmax(
                teacher_outputs['