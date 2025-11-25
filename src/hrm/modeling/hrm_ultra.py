"""
Ultra-Efficient Enhanced HRM for <500K params and >40% accuracy.

Simplifications for parameter efficiency:
- Standard attention (not multi-scale) with recency bias
- Standard location embeddings (not hierarchical)
- Optimized for maximum hidden size within budget

Keeps:
- Enhanced temporal embeddings with MLP
- Cross-attention
- Gated fusion
- EMA and advanced training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


@dataclass
class UltraConfig:
    max_seq_len: int = 50
    vocab_size: int = 1200
    hidden_size: int = 128  # Larger!
    num_layers: int = 2
    num_heads: int = 4
    expansion: float = 2.0
    high_level_cycles: int = 2
    low_level_cycles: int = 1
    use_features: bool = True
    num_users: int = 182
    dtype: torch.dtype = torch.float32
    dropout: float = 0.15
    recency_decay: float = 0.9


class EnhancedTemporalEmbedding(nn.Module):
    """Enhanced temporal with MLP interaction."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        embed_dim = hidden_size // 6
        self.hour_embed = nn.Embedding(24, embed_dim)
        self.minute_embed = nn.Embedding(60, embed_dim)
        self.weekday_embed = nn.Embedding(7, embed_dim)
        
        self.interaction_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
    
    def forward(self, hour, minute, weekday):
        h_emb = self.hour_embed(hour)
        m_emb = self.minute_embed(minute)
        w_emb = self.weekday_embed(weekday)
        concat = torch.cat([h_emb, m_emb, w_emb], dim=-1)
        return self.interaction_mlp(concat)


class RecencyBiasedAttention(nn.Module):
    """Standard attention with recency bias for efficiency."""
    
    def __init__(self, hidden_size: int, num_heads: int, recency_decay: float, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.recency_decay = recency_decay
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, D]
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Recency bias
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0) - torch.arange(seq_len, device=x.device).unsqueeze(1)
        bias = torch.where(
            positions <= 0,
            positions.float() * math.log(self.recency_decay),
            torch.zeros_like(positions, dtype=torch.float32)
        )
        attn = attn + bias.unsqueeze(0).unsqueeze(0)
        
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """Efficient cross-attention."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.kv_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value, mask=None):
        batch_size, query_len, _ = query.shape
        kv_len = key_value.shape[1]
        
        q = self.q_proj(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(key_value).reshape(batch_size, kv_len, 2, self.num_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, query_len, self.hidden_size)
        return self.out_proj(out)


class GatedFusion(nn.Module):
    """GRU-inspired gating."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.reset_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.update_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.candidate = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=-1)
        r = torch.sigmoid(self.reset_gate(combined))
        z = torch.sigmoid(self.update_gate(combined))
        reset_combined = torch.cat([r * x1, x2], dim=-1)
        h_candidate = torch.tanh(self.candidate(reset_combined))
        return (1 - z) * x1 + z * h_candidate


class EfficientTransformerBlock(nn.Module):
    """Efficient transformer with recency-biased attention."""
    
    def __init__(self, config: UltraConfig):
        super().__init__()
        self.attention = RecencyBiasedAttention(
            config.hidden_size,
            config.num_heads,
            config.recency_decay,
            config.dropout
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, int(config.hidden_size * config.expansion)),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(int(config.hidden_size * config.expansion), config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        
    def forward(self, x, mask=None):
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class UltraEfficientHRM(nn.Module):
    """Ultra-efficient HRM: <500K params, >40% target."""
    
    def __init__(self, config: UltraConfig, generator=None, device='cuda'):
        super().__init__()
        self.config = config
        self.device = device
        
        # Standard embeddings
        self.location_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.temporal_embed = EnhancedTemporalEmbedding(config.hidden_size)
        
        if config.use_features:
            self.user_embed = nn.Embedding(config.num_users, config.hidden_size // 4)
        self.duration_proj = nn.Linear(1, config.hidden_size // 4)
        
        input_dim = config.hidden_size * 2
        if config.use_features:
            input_dim += config.hidden_size // 2
        self.input_proj = nn.Linear(input_dim, config.hidden_size)
        
        # Reasoning state
        self.reasoning_init = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        
        # Transformer blocks
        self.high_level_blocks = nn.ModuleList([
            EfficientTransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.low_level_blocks = nn.ModuleList([
            EfficientTransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Cross-attention and fusion
        self.cross_attention = CrossAttention(config.hidden_size, config.num_heads, config.dropout)
        self.fusion_reasoning = GatedFusion(config.hidden_size)
        self.fusion_cross_attn = GatedFusion(config.hidden_size)
        
        # Output
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
        
        loc_emb = self.location_embed(location_ids)
        
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
        reasoning_state = self.reasoning_init.expand(batch_size, 1, -1)
        
        for _ in range(self.config.high_level_cycles):
            for block in self.high_level_blocks:
                x = block(x)
            
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


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
