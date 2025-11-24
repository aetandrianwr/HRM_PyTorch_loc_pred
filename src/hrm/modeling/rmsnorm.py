"""
RMSNorm (Root Mean Square Layer Normalization).
Faithful PyTorch conversion from RMSNorm.swift
"""

import torch
import torch.nn as nn


def rms_norm(x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Apply RMS normalization.
    
    Args:
        x: Input tensor
        epsilon: Small constant for numerical stability
        
    Returns:
        Normalized tensor
    """
    original_dtype = x.dtype
    x = x.float()
    
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    
    return x.to(original_dtype)


class RMSNorm(nn.Module):
    """RMSNorm module wrapper."""
    
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.epsilon)
