"""
Hierarchical Reasoning Model with Adaptive Computation Time.
Faithful PyTorch conversion from HRM.swift
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
import math

from .attention import Attention
from .swiglu import SwiGLU
from .rmsnorm import rms_norm
from .embedding import Embedding
from .linear import Linear
from .rotary import RotaryPositionEmbedding
from .init_utils import trunc_normal_init


@dataclass
class TransformerConfig:
    """Transformer configuration."""
    num_layers: int
    hidden_size: int
    num_heads: int
    expansion: float
    norm_epsilon: float = 1e-5
    rope_theta: float = 10000.0


@dataclass
class ACTConfig:
    """Adaptive Computation Time configuration."""
    halt_max_steps: int
    halt_exploration_probability: float


@dataclass
class HRMACTModelConfig:
    """HRM-ACT model configuration."""
    seq_len: int
    vocab_size: int
    high_level_cycles: int
    low_level_cycles: int
    transformers: TransformerConfig
    act: ACTConfig
    dtype: torch.dtype = torch.bfloat16


class HRMACTBlock(nn.Module):
    """Single HRM-ACT transformer block."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        norm_epsilon: float,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator = None,
        device: torch.device = None,
    ):
        super().__init__()
        
        self.norm_epsilon = norm_epsilon
        
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
        """
        Forward pass.
        
        Args:
            x: Input tensor
            rotary_position_embedding: Optional rotary embeddings
            
        Returns:
            Output tensor
        """
        # Self-attention with residual and RMSNorm
        x = rms_norm(
            x + self.self_attn(x, rotary_position_embedding),
            epsilon=self.norm_epsilon
        )
        
        # MLP with residual and RMSNorm
        x = rms_norm(x + self.mlp(x), epsilon=self.norm_epsilon)
        
        return x


class HRMACTReasoner(nn.Module):
    """HRM-ACT reasoning module with multiple transformer blocks."""
    
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        norm_epsilon: float,
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
                HRMACTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    expansion=expansion,
                    norm_epsilon=norm_epsilon,
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
        """
        Forward pass.
        
        Args:
            hidden_state: Current hidden state
            input_injection: Input to inject
            rotary_position_embedding: Optional rotary embeddings
            
        Returns:
            Updated hidden state
        """
        hidden_state = hidden_state + input_injection
        
        for block in self.blocks:
            hidden_state = block(hidden_state, rotary_position_embedding)
        
        return hidden_state


class HiddenStates:
    """Container for high-level and low-level hidden states."""
    
    def __init__(self, high_level: torch.Tensor, low_level: torch.Tensor):
        self.high_level = high_level
        self.low_level = low_level
    
    def map(self, func):
        """Apply function to both states."""
        return HiddenStates(
            high_level=func(self.high_level),
            low_level=func(self.low_level),
        )


class HRMACTOutput:
    """Output from HRM-ACT model."""
    
    def __init__(
        self,
        hidden_states: HiddenStates,
        output: torch.Tensor,
        q_act_halt: torch.Tensor,
        q_act_continue: torch.Tensor,
    ):
        self.hidden_states = hidden_states
        self.output = output
        self.q_act_halt = q_act_halt
        self.q_act_continue = q_act_continue


class HRMACTInner(nn.Module):
    """
    Inner HRM-ACT model - the core hierarchical reasoning module.
    """
    
    def __init__(
        self,
        config: HRMACTModelConfig,
        generator: torch.Generator,
        device: torch.device = None,
    ):
        super().__init__()
        
        self.config = config
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Create generator copies for different components
        gen = generator
        
        # CLS token
        gen_cls = torch.Generator(device=device)
        gen_cls.manual_seed(gen.initial_seed() + 1)
        self.cls_token = nn.Parameter(
            trunc_normal_init(
                (config.transformers.hidden_size,),
                std=1.0 / math.sqrt(config.transformers.hidden_size),
                dtype=config.dtype,
                generator=gen_cls,
                device=device,
            )
        )
        
        # Input embedding
        gen_emb = torch.Generator(device=device)
        gen_emb.manual_seed(gen.initial_seed() + 2)
        self.input_embedding = Embedding(
            vocab_size=config.vocab_size,
            dim=config.transformers.hidden_size,
            init_std=1.0 / math.sqrt(config.transformers.hidden_size),
            dtype=config.dtype,
            generator=gen_emb,
            device=device,
        )
        
        # Output head
        gen_out = torch.Generator(device=device)
        gen_out.manual_seed(gen.initial_seed() + 3)
        self.output_head = Linear(
            in_dim=config.transformers.hidden_size,
            out_dim=config.vocab_size,
            bias=False,
            dtype=config.dtype,
            generator=gen_out,
            device=device,
        )
        
        # Q-ACT head
        self.q_act_head = Linear(
            in_dim=config.transformers.hidden_size,
            out_dim=2,
            bias=True,
            dtype=config.dtype,
            device=device,
        )
        # Initialize with zeros for weight and -5 for bias
        nn.init.zeros_(self.q_act_head.weight)
        nn.init.constant_(self.q_act_head.bias, -5)
        
        # Rotary position embedding
        self.rotary_emb = RotaryPositionEmbedding(
            dim=config.transformers.hidden_size // config.transformers.num_heads,
            max_length=config.seq_len + 1,  # +1 for CLS token
            base=config.transformers.rope_theta,
            dtype=config.dtype,
            device=device,
        )
        
        # High-level reasoner
        gen_hl = torch.Generator(device=device)
        gen_hl.manual_seed(gen.initial_seed() + 4)
        self.high_level_reasoner = HRMACTReasoner(
            num_layers=config.transformers.num_layers,
            hidden_size=config.transformers.hidden_size,
            num_heads=config.transformers.num_heads,
            expansion=config.transformers.expansion,
            norm_epsilon=config.transformers.norm_epsilon,
            dtype=config.dtype,
            generator=gen_hl,
            device=device,
        )
        
        # Low-level reasoner
        gen_ll = torch.Generator(device=device)
        gen_ll.manual_seed(gen.initial_seed() + 5)
        self.low_level_reasoner = HRMACTReasoner(
            num_layers=config.transformers.num_layers,
            hidden_size=config.transformers.hidden_size,
            num_heads=config.transformers.num_heads,
            expansion=config.transformers.expansion,
            norm_epsilon=config.transformers.norm_epsilon,
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
                (config.transformers.hidden_size,),
                std=1.0,
                dtype=config.dtype,
                generator=gen_hl_init,
                device=device,
            )
        )
        self._initial_low_level = nn.Parameter(
            trunc_normal_init(
                (config.transformers.hidden_size,),
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
    def initial_hidden_states(self) -> HiddenStates:
        """Get initial hidden states."""
        return HiddenStates(
            high_level=self._initial_high_level,
            low_level=self._initial_low_level,
        )
    
    def forward(
        self,
        hidden_states: HiddenStates,
        inputs: torch.Tensor,
    ) -> HRMACTOutput:
        """
        Forward pass.
        
        Args:
            hidden_states: Current hidden states
            inputs: Input token indices [batch_size, seq_len]
            
        Returns:
            HRMACTOutput with predictions and updated states
        """
        batch_size = inputs.shape[0]
        
        # Create input embeddings with CLS token
        cls_tokens = self.cls_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        input_embs = self.input_embedding(inputs)
        input_embeddings = torch.cat([cls_tokens, input_embs], dim=1)
        input_embeddings = input_embeddings * math.sqrt(self.config.transformers.hidden_size)
        
        low_level_z = hidden_states.low_level
        high_level_z = hidden_states.high_level
        
        # Hierarchical reasoning cycles (all but last)
        total_cycles = self.config.high_level_cycles * self.config.low_level_cycles
        for cycle in range(1, total_cycles):
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
        
        # Stop gradients before final cycle
        low_level_z = low_level_z.detach()
        high_level_z = high_level_z.detach()
        
        # Final cycle with gradients
        low_level_z = self.low_level_reasoner(
            hidden_state=low_level_z,
            input_injection=high_level_z + input_embeddings,
            rotary_position_embedding=self.rotary_emb,
        )
        high_level_z = self.high_level_reasoner(
            hidden_state=high_level_z,
            input_injection=low_level_z,
            rotary_position_embedding=self.rotary_emb,
        )
        
        # Output predictions (excluding CLS token)
        output_logits = self.output_head(high_level_z[:, 1:])
        
        # Q-ACT predictions (CLS token only)
        q_act_logits = self.q_act_head(high_level_z[:, 0])
        
        return HRMACTOutput(
            hidden_states=HiddenStates(
                high_level=high_level_z.detach(),
                low_level=low_level_z.detach(),
            ),
            output=output_logits,
            q_act_halt=q_act_logits[:, 0],
            q_act_continue=q_act_logits[:, 1],
        )
