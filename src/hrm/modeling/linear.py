"""
Linear layer with custom initialization.
Faithful PyTorch conversion from Linear.swift
"""

import torch
import torch.nn as nn
from .init_utils import trunc_normal_init


class Linear(nn.Module):
    """
    Linear layer with truncated normal initialization.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator = None,
        device: torch.device = None,
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Initialize weight with truncated normal
        std = 1.0 / (in_dim ** 0.5)
        self.weight = nn.Parameter(
            trunc_normal_init(
                (in_dim, out_dim),
                std=std,
                dtype=dtype,
                generator=generator,
                device=device,
            )
        )
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim, dtype=dtype, device=device))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [..., in_dim]
            
        Returns:
            Output tensor [..., out_dim]
        """
        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x
