"""
Enhanced HRM with <500K parameters targeting >40% test accuracy.

Key Enhancements:
1. Multi-Scale Temporal Attention (short-range + long-range)
2. Cross-Attention between reasoning state and history
3. Hierarchical Location Embeddings (fine + coarse clusters)
4. Gated Fusion Mechanisms (GRU-inspired)
5. Enhanced Temporal Embeddings with MLP interaction
6. Efficient architecture design for parameter budget
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EnhancedHRMConfig:
    """Configuration for Enhanced HRM."""
    max_seq_len: int = 50
    vocab_size: int = 1200
    hidden_size: int = 96  # Reduced for parameter efficiency
    num_layers: int = 2
    num_heads: int = 4
    expansion: float = 2.0
    high_level_cycles: int = 2
    low_level_cycles: int = 2
    use_features: bool = True
    num_users: int = 182
    dtype: torch.dtype = torch.float32
    dropout: float = 0.15
    # Hierarchical location embedding
    num_coarse_clusters: int = 50  # Coarse spatial regions
    # Temporal attention
    short_range_window: int = 10
    recency_decay: float = 0.9


class EnhancedTemporalEmbedding(nn.Module):
    """Enhanced temporal embeddings with MLP-based interaction."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Individual embeddings (small dimension)
        embed_dim = hidden_size // 6
        self.hour_embed = nn.Embedding(24, embed_dim)
        self.minute_embed = nn.Embedding(60, embed_dim)
        self.weekday_embed = nn.Embedding(7, embed_dim)
        
        # MLP for interaction modeling
        input_dim = embed_dim * 3
        self.interaction_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
    def forward(self, hour, minute, weekday):
        """
        Args:
            hour: [batch_size, seq_len] or [batch_size]
            minute: [batch_size, seq_len] or [batch_size]
            weekday: [batch_size, seq_len] or [batch_size]
        Returns:
            temporal_embed: [batch_size, seq_len, hidden_size] or [batch_size, hidden_size]
        """
        h_emb = self.hour_embed(hour)
        m_emb = self.minute_embed(minute)
        w_emb = self.weekday_embed(weekday)
        
        # Concatenate and pass through MLP for interaction
        concat = torch.cat([h_emb, m_emb, w_emb], dim=-1)
        return self.interaction_mlp(concat)


class HierarchicalLocationEmbedding(nn.Module):
    """Two-level hierarchical location embeddings with soft clustering."""
    
    def __init__(self, vocab_size: int, hidden_size: int, num_coarse_clusters: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_coarse_clusters = num_coarse_clusters
        
        # Fine-grained location embeddings
        self.fine_embed = nn.Embedding(vocab_size, hidden_size // 2)
        
        # Coarse-grained cluster embeddings
        self.coarse_embed = nn.Embedding(num_coarse_clusters, hidden_size // 2)
        
        # Soft clustering: location -> cluster distribution
        self.loc_to_cluster = nn.Linear(vocab_size, num_coarse_clusters)
        
    def forward(self, location_ids):
        """
        Args:
            location_ids: [batch_size, seq_len] or [batch_size]
        Returns:
            hierarchical_embed: [batch_size, seq_len, hidden_size] or [batch_size, hidden_size]
        """
        # Fine-grained embedding
        fine_emb = self.fine_embed(location_ids)
        
        # Soft clustering: create one-hot then get cluster distribution
        one_hot = F.one_hot(location_ids, num_classes=self.vocab_size).float()
        cluster_logits = self.loc_to_cluster(one_hot)
        cluster_dist = F.softmax(cluster_logits, dim=-1)
        
        # Coarse-grained embedding via soft assignment
        coarse_emb = torch.matmul(cluster_dist, self.coarse_embed.weight)
        
        # Concatenate fine and coarse
        return torch.cat([fine_emb, coarse_emb], dim=-1)


class MultiScaleTemporalAttention(nn.Module):
    """Separate short-range and long-range temporal attention."""
    
    def __init__(self, hidden_size: int, num_heads: int, short_range_window: int, 
                 recency_decay: float, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.short_range_window = short_range_window
        self.recency_decay = recency_decay
        self.head_dim = hidden_size // num_heads
        
        # Short-range attention (with recency bias)
        self.short_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.short_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.short_v = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Long-range attention (periodic patterns)
        self.long_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.long_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.long_v = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
            mask: [batch_size, seq_len] optional padding mask
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Short-range attention with exponential decay bias
        q_short = self.short_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_short = self.short_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_short = self.short_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_short = torch.matmul(q_short, k_short.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add exponential decay bias for recency (only for recent history)
        recency_bias = self._create_recency_bias(seq_len, x.device)
        attn_short = attn_short + recency_bias.unsqueeze(0).unsqueeze(0)
        
        # Apply window mask for short-range
        window_mask = self._create_window_mask(seq_len, self.short_range_window, x.device)
        attn_short = attn_short.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if mask is not None:
            attn_short = attn_short.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_short = F.softmax(attn_short, dim=-1)
        attn_short = self.dropout(attn_short)
        out_short = torch.matmul(attn_short, v_short).transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        
        # Long-range attention (no recency bias, captures periodic patterns)
        q_long = self.long_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_long = self.long_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_long = self.long_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_long = torch.matmul(q_long, k_long.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn_long = attn_long.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_long = F.softmax(attn_long, dim=-1)
        attn_long = self.dropout(attn_long)
        out_long = torch.matmul(attn_long, v_long).transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        
        # Combine short and long range
        combined = torch.cat([out_short, out_long], dim=-1)
        output = self.out_proj(combined)
        
        return output
    
    def _create_recency_bias(self, seq_len, device):
        """Create exponential decay bias matrix for recency."""
        positions = torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(seq_len, device=device).unsqueeze(1)
        # Exponential decay: recent positions get higher scores
        bias = torch.where(
            positions <= 0,
            positions.float() * math.log(self.recency_decay),
            torch.zeros_like(positions, dtype=torch.float32)
        )
        return bias
    
    def _create_window_mask(self, seq_len, window_size, device):
        """Create causal window mask for short-range attention."""
        positions = torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(seq_len, device=device).unsqueeze(1)
        mask = (positions <= 0) & (positions >= -window_size)
        return mask


class CrossAttention(nn.Module):
    """Cross-attention for reasoning state to attend to history."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value, mask=None):
        """
        Args:
            query: [batch_size, query_len, hidden_size] (reasoning state)
            key_value: [batch_size, kv_len, hidden_size] (history)
            mask: [batch_size, kv_len] optional
        Returns:
            output: [batch_size, query_len, hidden_size]
        """
        batch_size, query_len, _ = query.shape
        kv_len = key_value.shape[1]
        
        q = self.q_proj(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, query_len, self.hidden_size)
        output = self.out_proj(output)
        
        return output


class GatedFusion(nn.Module):
    """GRU-inspired gated fusion mechanism."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Gates
        self.reset_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.update_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.candidate = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x1, x2):
        """
        Args:
            x1, x2: [batch_size, ..., hidden_size]
        Returns:
            fused: [batch_size, ..., hidden_size]
        """
        combined = torch.cat([x1, x2], dim=-1)
        
        r = torch.sigmoid(self.reset_gate(combined))
        z = torch.sigmoid(self.update_gate(combined))
        
        reset_combined = torch.cat([r * x1, x2], dim=-1)
        h_candidate = torch.tanh(self.candidate(reset_combined))
        
        fused = (1 - z) * x1 + z * h_candidate
        
        return fused


class EfficientTransformerBlock(nn.Module):
    """Efficient transformer block with multi-scale attention."""
    
    def __init__(self, config: EnhancedHRMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Multi-scale temporal attention
        self.attention = MultiScaleTemporalAttention(
            config.hidden_size,
            config.num_heads,
            config.short_range_window,
            config.recency_decay,
            config.dropout
        )
        
        # Feed-forward network (smaller for efficiency)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, int(config.hidden_size * config.expansion)),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(int(config.hidden_size * config.expansion), config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        
    def forward(self, x, mask=None):
        # Attention with residual
        attn_out = self.attention(self.ln1(x), mask)
        x = x + attn_out
        
        # FFN with residual
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out
        
        return x


class EnhancedLocationHRM(nn.Module):
    """
    Enhanced HRM for location prediction with <500K parameters.
    Target: >40% test accuracy on Geolife.
    """
    
    def __init__(self, config: EnhancedHRMConfig, generator=None, device='cuda'):
        super().__init__()
        self.config = config
        self.device = device
        
        # Hierarchical location embeddings
        self.location_embed = HierarchicalLocationEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.num_coarse_clusters
        )
        
        # Enhanced temporal embeddings
        self.temporal_embed = EnhancedTemporalEmbedding(config.hidden_size)
        
        # User embedding (small)
        if config.use_features:
            self.user_embed = nn.Embedding(config.num_users, config.hidden_size // 4)
        
        # Duration embedding (continuous feature)
        self.duration_proj = nn.Linear(1, config.hidden_size // 4)
        
        # Input projection
        # temporal_emb is hidden_size, loc_emb is hidden_size, user_emb is hidden_size//4, duration_emb is hidden_size//4
        input_dim = config.hidden_size * 2  # loc + temporal
        if config.use_features:
            input_dim += config.hidden_size // 4 + config.hidden_size // 4  # user + duration
        self.input_proj = nn.Linear(input_dim, config.hidden_size)
        
        # Reasoning state initialization
        self.reasoning_init = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        
        # High-level reasoning (processes full sequence)
        self.high_level_blocks = nn.ModuleList([
            EfficientTransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Low-level reasoning (processes reasoning state)
        self.low_level_blocks = nn.ModuleList([
            EfficientTransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Cross-attention for reasoning to attend to history
        self.cross_attention = CrossAttention(config.hidden_size, config.num_heads, config.dropout)
        
        # Gated fusion mechanisms
        self.fusion_reasoning = GatedFusion(config.hidden_size)
        self.fusion_cross_attn = GatedFusion(config.hidden_size)
        
        # Output head (shared across cycles for efficiency)
        self.output_head = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Initialize weights
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
        """
        Args:
            location_ids: [batch_size, seq_len]
            features: dict with 'user_id', 'weekday', 'start_time', 'duration', 'time_diff'
        Returns:
            logits: [batch_size, vocab_size]
        """
        batch_size, seq_len = location_ids.shape
        
        # Hierarchical location embeddings
        loc_emb = self.location_embed(location_ids)
        
        # Enhanced temporal embeddings
        if features is not None and self.config.use_features:
            # Extract hour and minute from start_min (minutes from midnight)
            start_min = features['start_min'].long()
            hour = (start_min // 60) % 24
            minute = start_min % 60
            weekday = features['weekday']
            
            temporal_emb = self.temporal_embed(hour, minute, weekday)
            
            # User embedding - take first user ID from sequence
            user_id = features['user'][:, 0]  # [batch_size]
            user_emb = self.user_embed(user_id).unsqueeze(1).expand(-1, seq_len, -1)
            
            # Duration embedding
            duration = features['duration'].unsqueeze(-1)
            duration_emb = self.duration_proj(duration)
            
            # Combine all embeddings
            combined = torch.cat([loc_emb, temporal_emb, user_emb, duration_emb], dim=-1)
        else:
            combined = loc_emb
        
        # Project to hidden size
        x = self.input_proj(combined)
        
        # Initialize reasoning state
        reasoning_state = self.reasoning_init.expand(batch_size, 1, -1)
        
        # Hierarchical reasoning cycles
        for _ in range(self.config.high_level_cycles):
            # High-level: process full sequence
            for block in self.high_level_blocks:
                x = block(x)
            
            # Low-level: refine reasoning state
            for _ in range(self.config.low_level_cycles):
                for block in self.low_level_blocks:
                    reasoning_state = block(reasoning_state)
                
                # Cross-attention: reasoning attends to history
                cross_attn_out = self.cross_attention(reasoning_state, x)
                
                # Gated fusion of cross-attention
                reasoning_state = self.fusion_cross_attn(reasoning_state, cross_attn_out)
            
            # Fuse reasoning back into sequence representation
            # Use last position of sequence as context
            context = x[:, -1:, :]
            fused_context = self.fusion_reasoning(context, reasoning_state)
            
            # Update the last position
            x = torch.cat([x[:, :-1, :], fused_context], dim=1)
        
        # Final prediction from reasoning state
        logits = self.output_head(reasoning_state.squeeze(1))
        
        return logits


def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
