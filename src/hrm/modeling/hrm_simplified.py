"""
Simplified Enhanced HRM - removes hierarchical embeddings to save parameters.
Keeps all other enhancements to maximize capacity within 500K budget.
"""

import torch
import torch.nn as nn
from .hrm_enhanced import (
    EnhancedTemporalEmbedding,
    MultiScaleTemporalAttention,
    CrossAttention,
    GatedFusion,
    EfficientTransformerBlock,
    EnhancedHRMConfig
)


class SimplifiedEnhancedHRM(nn.Module):
    """
    Simplified Enhanced HRM with standard embeddings for <500K params.
    Optimizes for larger hidden size by using simple location embeddings.
    """
    
    def __init__(self, config: EnhancedHRMConfig, generator=None, device='cuda'):
        super().__init__()
        self.config = config
        self.device = device
        
        # Simple location embedding (no hierarchy to save params)
        self.location_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Enhanced temporal embeddings
        self.temporal_embed = EnhancedTemporalEmbedding(config.hidden_size)
        
        # User embedding (small)
        if config.use_features:
            self.user_embed = nn.Embedding(config.num_users, config.hidden_size // 4)
        
        # Duration embedding
        self.duration_proj = nn.Linear(1, config.hidden_size // 4)
        
        # Input projection
        input_dim = config.hidden_size * 2  # loc + temporal
        if config.use_features:
            input_dim += config.hidden_size // 4 + config.hidden_size // 4
        self.input_proj = nn.Linear(input_dim, config.hidden_size)
        
        # Reasoning state
        self.reasoning_init = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        
        # High-level reasoning
        self.high_level_blocks = nn.ModuleList([
            EfficientTransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Low-level reasoning
        self.low_level_blocks = nn.ModuleList([
            EfficientTransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Cross-attention
        self.cross_attention = CrossAttention(config.hidden_size, config.num_heads, config.dropout)
        
        # Gated fusion
        self.fusion_reasoning = GatedFusion(config.hidden_size)
        self.fusion_cross_attn = GatedFusion(config.hidden_size)
        
        # Output head
        self.output_head = nn.Linear(config.hidden_size, config.vocab_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, location_ids, features=None):
        batch_size, seq_len = location_ids.shape
        
        # Standard location embeddings
        loc_emb = self.location_embed(location_ids)
        
        # Enhanced temporal embeddings
        if features is not None and self.config.use_features:
            start_min = features['start_min'].long()
            hour = torch.div(start_min, 60, rounding_mode='floor') % 24
            minute = start_min % 60
            weekday = features['weekday']
            
            temporal_emb = self.temporal_embed(hour, minute, weekday)
            
            user_id = features['user'][:, 0]
            user_emb = self.user_embed(user_id).unsqueeze(1).expand(-1, seq_len, -1)
            
            duration = features['duration'].unsqueeze(-1)
            duration_emb = self.duration_proj(duration)
            
            combined = torch.cat([loc_emb, temporal_emb, user_emb, duration_emb], dim=-1)
        else:
            combined = loc_emb
        
        x = self.input_proj(combined)
        
        # Initialize reasoning state
        reasoning_state = self.reasoning_init.expand(batch_size, 1, -1)
        
        # Hierarchical reasoning
        for _ in range(self.config.high_level_cycles):
            # High-level
            for block in self.high_level_blocks:
                x = block(x)
            
            # Low-level
            for _ in range(self.config.low_level_cycles):
                for block in self.low_level_blocks:
                    reasoning_state = block(reasoning_state)
                
                cross_attn_out = self.cross_attention(reasoning_state, x)
                reasoning_state = self.fusion_cross_attn(reasoning_state, cross_attn_out)
            
            context = x[:, -1:, :]
            fused_context = self.fusion_reasoning(context, reasoning_state)
            x = torch.cat([x[:, :-1, :], fused_context], dim=1)
        
        logits = self.output_head(reasoning_state.squeeze(1))
        return logits
