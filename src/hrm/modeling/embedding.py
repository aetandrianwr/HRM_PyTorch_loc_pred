"""
Embedding layer with custom initialization.
Faithful PyTorch conversion from Embedding.swift
"""

import torch
import torch.nn as nn
from .init_utils import trunc_normal_init


class Embedding(nn.Module):
    """
    Embedding layer with truncated normal initialization.
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        init_std: float,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator = None,
        device: torch.device = None,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim = dim
        
        # Initialize embeddings with truncated normal
        self.embeddings = nn.Parameter(
            trunc_normal_init(
                (vocab_size, dim),
                std=init_std,
                dtype=dtype,
                generator=generator,
                device=device,
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - retrieve embeddings for input indices.
        
        Args:
            x: Input tensor of indices [..., seq_len]
            
        Returns:
            Embedded tensor [..., seq_len, dim]
        """
        return self.embeddings[x]
