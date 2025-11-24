"""
Multi-head attention mechanism.
Faithful PyTorch conversion from Attention.swift
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear import Linear
from .rotary import RotaryPositionEmbedding


class Attention(nn.Module):
    """
    Multi-head attention with optional rotary position embeddings.
    """
    
    def __init__(
        self,
        dim: int,
        head_dim: int,
        num_heads: int,
        key_value_heads_per_head: int,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator = None,
        device: torch.device = None,
    ):
        super().__init__()
        
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.output_size = head_dim * num_heads
        self.key_value_heads_per_head = key_value_heads_per_head
        self.num_key_value_heads = num_heads * key_value_heads_per_head
        
        # Split generator for two linear layers
        if generator is not None:
            gen1 = torch.Generator(device=generator.device if hasattr(generator, 'device') else None)
            gen2 = torch.Generator(device=generator.device if hasattr(generator, 'device') else None)
            gen1.manual_seed(generator.initial_seed())
            gen2.manual_seed(generator.initial_seed() + 1)
        else:
            gen1, gen2 = None, None
        
        # QKV projection
        self.qkv_proj = Linear(
            dim,
            (num_heads + 2 * self.num_key_value_heads) * head_dim,
            bias=False,
            dtype=dtype,
            generator=gen1,
            device=device,
        )
        
        # Output projection
        self.out_proj = Linear(
            self.output_size,
            dim,
            bias=False,
            dtype=dtype,
            generator=gen2,
            device=device,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        rotary_position_embedding: RotaryPositionEmbedding = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            rotary_position_embedding: Optional rotary position embedding
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x).view(
            batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim
        )
        
        # Split into Q, K, V
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads:self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        
        # Apply rotary position embedding if provided
        if rotary_position_embedding is not None:
            query = rotary_position_embedding(query)
            key = rotary_position_embedding(key)
        
        # Reshape query: [batch, seq_len, num_kv_heads, kv_heads_per_head, head_dim]
        # -> [batch, num_kv_heads, kv_heads_per_head, seq_len, head_dim]
        query = query.view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.key_value_heads_per_head,
            self.head_dim,
        ).permute(0, 2, 3, 1, 4)
        
        # Reshape key: [batch, seq_len, num_kv_heads, head_dim]
        # -> [batch, num_kv_heads, 1, head_dim, seq_len]
        key = key.permute(0, 2, 3, 1).unsqueeze(2)
        
        # Reshape value: [batch, seq_len, num_kv_heads, head_dim]
        # -> [batch, num_kv_heads, 1, seq_len, head_dim]
        value = value.permute(0, 2, 1, 3).unsqueeze(2)
        
        # Compute attention logits
        attn_logits = torch.matmul(query, key) * (1.0 / (self.head_dim ** 0.5))
        
        # Apply softmax
        attn_weights = F.softmax(attn_logits.float(), dim=-1)

        # Apply attention to values (ensure dtype consistency)
        combined = torch.matmul(attn_weights, value.float()).to(value.dtype)
        
        # Reshape back: [batch, num_kv_heads, kv_heads_per_head, seq_len, head_dim]
        # -> [batch, seq_len, num_kv_heads, kv_heads_per_head, head_dim]
        # -> [batch, seq_len, dim]
        combined = combined.permute(0, 3, 1, 2, 4).contiguous().view(
            batch_size, seq_len, self.dim
        )
        
        # Apply output projection
        return self.out_proj(combined)
