"""
SwiGLU activation layer.
Faithful PyTorch conversion from SwiGLU.swift
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear import Linear


class SwiGLU(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) activation layer.
    """
    
    def __init__(
        self,
        dim: int,
        expansion: float,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator = None,
        device: torch.device = None,
    ):
        super().__init__()
        
        # Compute intermediate dimension
        inter_dim = self._find_multiple(int(expansion * dim * 2.0 / 3.0), 256)
        
        # Split generator for two linear layers
        if generator is not None:
            # Create two generators with different states
            gen1 = torch.Generator(device=generator.device if hasattr(generator, 'device') else None)
            gen2 = torch.Generator(device=generator.device if hasattr(generator, 'device') else None)
            gen1.manual_seed(generator.initial_seed())
            gen2.manual_seed(generator.initial_seed() + 1)
        else:
            gen1, gen2 = None, None
        
        self.gate_up_proj = Linear(
            dim, inter_dim * 2, bias=False, dtype=dtype, generator=gen1, device=device
        )
        self.down_proj = Linear(
            inter_dim, dim, bias=False, dtype=dtype, generator=gen2, device=device
        )
    
    @staticmethod
    def _find_multiple(a: int, b: int) -> int:
        """Find the smallest multiple of b that is >= a."""
        return (-(a // -b)) * b
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [..., dim]
            
        Returns:
            Output tensor [..., dim]
        """
        # Split gate and up projections
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        
        # Apply SiLU (Swish) activation to gate and multiply with up
        return self.down_proj(F.silu(gate) * up)
