"""
Rotary Position Embedding (RoPE).
Faithful PyTorch conversion from RotaryPositionEmbedding.swift
"""

import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding for positional encoding.
    """
    
    def __init__(
        self,
        dim: int,
        max_length: int,
        base: float,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        super().__init__()
        
        self.dim = dim
        self.max_length = max_length
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=dtype, device=device).float() / dim)
        )
        
        # Compute position indices
        t = torch.arange(max_length, dtype=dtype, device=device)
        
        # Compute frequencies
        freqs = torch.outer(t, inv_freq)
        
        # Concatenate and expand dimensions
        emb = torch.cat([freqs, freqs], dim=-1).unsqueeze(-2)
        
        # Compute cos and sin
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)
    
    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the hidden dims of the input.
        
        Args:
            x: Input tensor
            
        Returns:
            Rotated tensor
        """
        # Move last dimension to first
        x = x.movedim(-1, 0)
        half_dim = x.shape[0] // 2
        x1 = x[:half_dim]
        x2 = x[half_dim:]
        # Concatenate [-x2, x1] and move back
        return torch.cat([-x2, x1], dim=0).movedim(0, -1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embedding.
        
        Args:
            x: Input tensor [..., seq_len, ..., dim]
            
        Returns:
            Tensor with rotary embeddings applied
        """
        # Get sequence length from input
        seq_len = x.shape[1]
        # Slice cos and sin to match sequence length
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]
        return (x * cos) + (self.rotate_half(x) * sin)
