"""
Hierarchical Reasoning Model for Next Location Prediction.
Adapted from HRM-ACT for sequential location prediction.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math

from .attention import Attention
from .swiglu import SwiGLU
from .rmsnorm import rms_norm
from .embedding import Embedding
from .linear import Linear
from .rotary import RotaryPositionEmbedding
from .init_utils import trunc_normal_init


@dataclass
class LocationHRMConfig:
    """Configuration for Location HRM."""
    max_seq_len: int
    vocab_size: int
    high_level_cycles: int
    low_level_cycles: int
    hidden_size: int
    num_layers: int
    num_heads: int
    expansion: float
    norm_epsilon: float = 1e-5
    rope_theta: float = 10000.0
    dtype: torch.dtype = torch.bfloat16
    use_features: bool = True
    # Feature vocabulary sizes
    num_users: int = 100
    num_weekdays: int = 7
    feature_dim: int = 32
    dropout: float = 0.1
    max_start_min: int = 1440  # minutes in a day


class LocationHiddenStates:
    """Container for hierarchical hidden states."""
    
    def __init__(self, high_level: torch.Tensor, low_level: torch.Tensor):
        self.high_level = high_level
        self.low_level = low_level
    
    def detach(self):
        """Detach both states."""
        return LocationHiddenStates(
            high_level=self.high_level.detach(),
            low_level=self.low_level.detach(),
        )


class LocationHRMBlock(nn.Module):
    """Single HRM transformer block."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        norm_epsilon: float,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator = None,
        device: torch.device = None,
    ):
        super().__init__()
        
        self.norm_epsilon = norm_epsilon
        self.dropout = nn.Dropout(dropout)
        
        # Split generator
        if generator is not None:
            gen1 = torch.Generator(device=generator.device if hasattr(generator, 'device') else None)
            gen2 = torch.Generator(device=generator.device if hasattr(generator, 'device') else None)
            gen1.manual_seed(generator.initial_seed())
            gen2.manual_seed(generator.initial_seed() + 1)
        else:
            gen1, gen2 = None, None
        
        self.self_attn = Attention(
            dim=hidden_size,
            head_dim=hidden_size // num_heads,
            num_heads=num_heads,
            key_value_heads_per_head=1,
            dtype=dtype,
            generator=gen1,
            device=device,
        )
        
        self.mlp = SwiGLU(
            dim=hidden_size,
            expansion=expansion,
            dtype=dtype,
            generator=gen2,
            device=device,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        rotary_position_embedding: Optional[RotaryPositionEmbedding] = None,
    ) -> torch.Tensor:
        # Self-attention with residual and RMSNorm
        attn_out = self.self_attn(x, rotary_position_embedding)
        attn_out = self.dropout(attn_out)
        x = rms_norm(x + attn_out, epsilon=self.norm_epsilon)
        
        # MLP with residual and RMSNorm
        mlp_out = self.mlp(x)
        mlp_out = self.dropout(mlp_out)
        x = rms_norm(x + mlp_out, epsilon=self.norm_epsilon)
        
        return x


class LocationHRMReasoner(nn.Module):
    """HRM reasoning module with multiple transformer blocks."""
    
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        norm_epsilon: float,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator = None,
        device: torch.device = None,
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        
        for i in range(num_layers):
            if generator is not None:
                gen = torch.Generator(device=generator.device if hasattr(generator, 'device') else None)
                gen.manual_seed(generator.initial_seed() + i)
            else:
                gen = None
            
            self.blocks.append(
                LocationHRMBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    expansion=expansion,
                    norm_epsilon=norm_epsilon,
                    dropout=dropout,
                    dtype=dtype,
                    generator=gen,
                    device=device,
                )
            )
    
    def forward(
        self,
        hidden_state: torch.Tensor,
        input_injection: torch.Tensor,
        rotary_position_embedding: Optional[RotaryPositionEmbedding] = None,
    ) -> torch.Tensor:
        hidden_state = hidden_state + input_injection
        
        for block in self.blocks:
            hidden_state = block(hidden_state, rotary_position_embedding)
        
        return hidden_state


class LocationHRM(nn.Module):
    """
    Hierarchical Reasoning Model for Next Location Prediction.
    """
    
    def __init__(
        self,
        config: LocationHRMConfig,
        generator: torch.Generator,
        device: torch.device = None,
    ):
        super().__init__()
        
        self.config = config
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        gen = generator
        
        # CLS token for sequence aggregation
        gen_cls = torch.Generator(device=device)
        gen_cls.manual_seed(gen.initial_seed() + 1)
        self.cls_token = nn.Parameter(
            trunc_normal_init(
                (config.hidden_size,),
                std=1.0 / math.sqrt(config.hidden_size),
                dtype=config.dtype,
                generator=gen_cls,
                device=device,
            )
        )
        
        # Location embedding
        gen_loc = torch.Generator(device=device)
        gen_loc.manual_seed(gen.initial_seed() + 2)
        self.location_embedding = Embedding(
            vocab_size=config.vocab_size,
            dim=config.hidden_size,
            init_std=1.0 / math.sqrt(config.hidden_size),
            dtype=config.dtype,
            generator=gen_loc,
            device=device,
        )
        
        # Feature embeddings
        if config.use_features:
            self.user_embedding = Embedding(
                vocab_size=config.num_users,
                dim=config.feature_dim,
                init_std=1.0 / math.sqrt(config.feature_dim),
                dtype=config.dtype,
                device=device,
            )
            self.weekday_embedding = Embedding(
                vocab_size=config.num_weekdays,
                dim=config.feature_dim,
                init_std=1.0 / math.sqrt(config.feature_dim),
                dtype=config.dtype,
                device=device,
            )
            # Linear projections for continuous features
            self.time_proj = Linear(
                in_dim=1,
                out_dim=config.feature_dim,
                bias=True,
                dtype=config.dtype,
                device=device,
            )
            self.duration_proj = Linear(
                in_dim=1,
                out_dim=config.feature_dim,
                bias=True,
                dtype=config.dtype,
                device=device,
            )
            self.diff_proj = Linear(
                in_dim=1,
                out_dim=config.feature_dim,
                bias=True,
                dtype=config.dtype,
                device=device,
            )
            
            # Combine features with location embedding
            feature_total_dim = config.feature_dim * 5  # user, weekday, time, duration, diff
            self.feature_combine = Linear(
                in_dim=config.hidden_size + feature_total_dim,
                out_dim=config.hidden_size,
                bias=True,
                dtype=config.dtype,
                device=device,
            )
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Output prediction head
        gen_out = torch.Generator(device=device)
        gen_out.manual_seed(gen.initial_seed() + 3)
        self.output_head = Linear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            bias=False,
            dtype=config.dtype,
            generator=gen_out,
            device=device,
        )
        
        # Rotary position embedding
        self.rotary_emb = RotaryPositionEmbedding(
            dim=config.hidden_size // config.num_heads,
            max_length=config.max_seq_len + 1,  # +1 for CLS
            base=config.rope_theta,
            dtype=config.dtype,
            device=device,
        )
        
        # High-level reasoner
        gen_hl = torch.Generator(device=device)
        gen_hl.manual_seed(gen.initial_seed() + 4)
        self.high_level_reasoner = LocationHRMReasoner(
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expansion=config.expansion,
            norm_epsilon=config.norm_epsilon,
            dropout=config.dropout,
            dtype=config.dtype,
            generator=gen_hl,
            device=device,
        )
        
        # Low-level reasoner
        gen_ll = torch.Generator(device=device)
        gen_ll.manual_seed(gen.initial_seed() + 5)
        self.low_level_reasoner = LocationHRMReasoner(
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expansion=config.expansion,
            norm_epsilon=config.norm_epsilon,
            dropout=config.dropout,
            dtype=config.dtype,
            generator=gen_ll,
            device=device,
        )
        
        # Initial hidden states
        gen_init = torch.Generator(device=device)
        gen_init.manual_seed(gen.initial_seed() + 6)
        gen_hl_init = torch.Generator(device=device)
        gen_hl_init.manual_seed(gen_init.initial_seed())
        gen_ll_init = torch.Generator(device=device)
        gen_ll_init.manual_seed(gen_init.initial_seed() + 1)
        
        self._initial_high_level = nn.Parameter(
            trunc_normal_init(
                (config.hidden_size,),
                std=1.0,
                dtype=config.dtype,
                generator=gen_hl_init,
                device=device,
            )
        )
        self._initial_low_level = nn.Parameter(
            trunc_normal_init(
                (config.hidden_size,),
                std=1.0,
                dtype=config.dtype,
                generator=gen_ll_init,
                device=device,
            )
        )
        
        # Freeze initial states
        self._initial_high_level.requires_grad = False
        self._initial_low_level.requires_grad = False
    
    @property
    def initial_hidden_states(self) -> LocationHiddenStates:
        """Get initial hidden states."""
        return LocationHiddenStates(
            high_level=self._initial_high_level,
            low_level=self._initial_low_level,
        )
    
    def forward(
        self,
        inputs: torch.Tensor,
        features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass for next location prediction.
        
        Args:
            inputs: Input location IDs [batch_size, seq_len]
            features: Optional dict with additional features
            
        Returns:
            Output logits [batch_size, vocab_size]
        """
        batch_size, seq_len = inputs.shape
        
        # Create embeddings
        loc_embs = self.location_embedding(inputs)
        
        # Add features if available
        if self.config.use_features and features is not None:
            user_emb = self.user_embedding(features['user'])
            weekday_emb = self.weekday_embedding(features['weekday'])
            time_emb = self.time_proj(features['start_min'].unsqueeze(-1) / self.config.max_start_min)
            dur_emb = self.duration_proj(features['duration'].unsqueeze(-1) / 1000.0)  # normalize
            diff_emb = self.diff_proj(features['diff'].unsqueeze(-1).float() / 10.0)  # normalize
            
            # Concatenate and project
            combined = torch.cat([
                loc_embs,
                user_emb,
                weekday_emb,
                time_emb,
                dur_emb,
                diff_emb,
            ], dim=-1)
            input_embeddings = self.feature_combine(combined)
        else:
            input_embeddings = loc_embs
        
        # Scale embeddings
        input_embeddings = input_embeddings * math.sqrt(self.config.hidden_size)
        input_embeddings = self.dropout(input_embeddings)
        
        # Add CLS token
        cls_tokens = self.cls_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        input_embeddings = torch.cat([cls_tokens, input_embeddings], dim=1)
        
        # Initialize hidden states
        low_level_z = self._initial_low_level.unsqueeze(0).expand(batch_size, seq_len + 1, -1)
        high_level_z = self._initial_high_level.unsqueeze(0).expand(batch_size, seq_len + 1, -1)
        
        # Hierarchical reasoning cycles
        total_cycles = self.config.high_level_cycles * self.config.low_level_cycles
        
        for cycle in range(1, total_cycles + 1):
            low_level_z = self.low_level_reasoner(
                hidden_state=low_level_z,
                input_injection=high_level_z + input_embeddings,
                rotary_position_embedding=self.rotary_emb,
            )
            
            if cycle % self.config.low_level_cycles == 0:
                high_level_z = self.high_level_reasoner(
                    hidden_state=high_level_z,
                    input_injection=low_level_z,
                    rotary_position_embedding=self.rotary_emb,
                )
        
        # Use CLS token for prediction
        cls_output = high_level_z[:, 0]
        
        # Output predictions
        output_logits = self.output_head(cls_output)
        
        return output_logits
