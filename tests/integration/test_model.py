"""
Integration tests for model components.
"""

import pytest
import torch
from src.hrm.modeling import (
    HRMACTInner,
    HRMACTModelConfig,
    TransformerConfig,
    ACTConfig,
    HiddenStates,
)


class TestModelIntegration:
    """Test integration of model components."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return HRMACTModelConfig(
            seq_len=81,
            vocab_size=10,
            high_level_cycles=2,
            low_level_cycles=2,
            transformers=TransformerConfig(
                num_layers=2,
                hidden_size=64,
                num_heads=4,
                expansion=2.0,
            ),
            act=ACTConfig(
                halt_max_steps=4,
                halt_exploration_probability=0.1,
            ),
            dtype=torch.float32,
        )
    
    @pytest.fixture
    def model(self, config):
        """Create test model."""
        generator = torch.Generator()
        generator.manual_seed(42)
        device = torch.device('cpu')
        return HRMACTInner(config=config, generator=generator, device=device)
    
    def test_model_initialization(self, model, config):
        """Test that model initializes correctly."""
        assert model.config == config
        assert model.cls_token.shape == (config.transformers.hidden_size,)
        assert model.input_embedding.vocab_size == config.vocab_size
    
    def test_initial_hidden_states(self, model, config):
        """Test initial hidden states."""
        states = model.initial_hidden_states
        
        assert states.high_level.shape == (config.transformers.hidden_size,)
        assert states.low_level.shape == (config.transformers.hidden_size,)
    
    def test_forward_pass_shape(self, model, config):
        """Test forward pass output shapes."""
        batch_size = 4
        inputs = torch.randint(0, config.vocab_size, (batch_size, config.seq_len))
        
        states = HiddenStates(
            high_level=model.initial_hidden_states.high_level.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1),
            low_level=model.initial_hidden_states.low_level.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1),
        )
        
        output = model(hidden_states=states, inputs=inputs)
        
        assert output.output.shape == (batch_size, config.seq_len, config.vocab_size)
        assert output.q_act_halt.shape == (batch_size,)
        assert output.q_act_continue.shape == (batch_size,)
        assert output.hidden_states.high_level.shape == (batch_size, config.seq_len + 1, config.transformers.hidden_size)
        assert output.hidden_states.low_level.shape == (batch_size, config.seq_len + 1, config.transformers.hidden_size)
    
    def test_forward_pass_no_nan(self, model, config):
        """Test that forward pass produces no NaN values."""
        batch_size = 2
        inputs = torch.randint(0, config.vocab_size, (batch_size, config.seq_len))
        
        states = HiddenStates(
            high_level=model.initial_hidden_states.high_level.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1),
            low_level=model.initial_hidden_states.low_level.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1),
        )
        
        output = model(hidden_states=states, inputs=inputs)
        
        assert not torch.any(torch.isnan(output.output))
        assert not torch.any(torch.isnan(output.q_act_halt))
        assert not torch.any(torch.isnan(output.q_act_continue))
    
    def test_backward_pass(self, model, config):
        """Test backward pass."""
        batch_size = 2
        inputs = torch.randint(0, config.vocab_size, (batch_size, config.seq_len))
        
        states = HiddenStates(
            high_level=model.initial_hidden_states.high_level.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1),
            low_level=model.initial_hidden_states.low_level.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1),
        )
        
        output = model(hidden_states=states, inputs=inputs)
        loss = output.output.sum() + output.q_act_halt.sum() + output.q_act_continue.sum()
        loss.backward()
        
        # Check that some parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.requires_grad:
                has_grad = True
                assert not torch.any(torch.isnan(param.grad))
        
        assert has_grad
    
    def test_model_reproducibility(self, config):
        """Test that same seed produces same model."""
        gen1 = torch.Generator()
        gen1.manual_seed(42)
        model1 = HRMACTInner(config=config, generator=gen1, device=torch.device('cpu'))
        
        gen2 = torch.Generator()
        gen2.manual_seed(42)
        model2 = HRMACTInner(config=config, generator=gen2, device=torch.device('cpu'))
        
        # Compare parameters
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert name1 == name2
            assert torch.allclose(param1, param2)
    
    def test_hierarchical_cycles(self, model, config):
        """Test that hierarchical reasoning cycles execute."""
        batch_size = 1
        inputs = torch.randint(0, config.vocab_size, (batch_size, config.seq_len))
        
        states = HiddenStates(
            high_level=model.initial_hidden_states.high_level.unsqueeze(0).unsqueeze(0),
            low_level=model.initial_hidden_states.low_level.unsqueeze(0).unsqueeze(0),
        )
        
        # Should complete without error
        output = model(hidden_states=states, inputs=inputs)
        
        # Hidden states should be different from initial
        assert not torch.allclose(
            output.hidden_states.high_level,
            states.high_level.expand_as(output.hidden_states.high_level)
        )
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_execution(self, config):
        """Test model execution on GPU."""
        device = torch.device('cuda')
        generator = torch.Generator(device=device)
        generator.manual_seed(42)
        
        model = HRMACTInner(config=config, generator=generator, device=device)
        model = model.to(device)
        
        batch_size = 2
        inputs = torch.randint(0, config.vocab_size, (batch_size, config.seq_len), device=device)
        
        states = HiddenStates(
            high_level=model.initial_hidden_states.high_level.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1),
            low_level=model.initial_hidden_states.low_level.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1),
        )
        
        output = model(hidden_states=states, inputs=inputs)
        
        assert output.output.device.type == 'cuda'
        assert output.q_act_halt.device.type == 'cuda'
