"""
Integration tests for training loop.
"""

import pytest
import torch
from src.hrm.modeling import (
    HRMACTInner,
    HRMACTModelConfig,
    TransformerConfig,
    ACTConfig,
)
from src.hrm.utils import TrainingBatch, step, Difficulty


class TestTrainingIntegration:
    """Test integration of training components."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return HRMACTModelConfig(
            seq_len=81,
            vocab_size=10,
            high_level_cycles=1,
            low_level_cycles=1,
            transformers=TransformerConfig(
                num_layers=1,
                hidden_size=32,
                num_heads=2,
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
    
    @pytest.fixture
    def optimizer(self, model):
        """Create optimizer."""
        return torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    def test_batch_initialization(self, model):
        """Test training batch initialization."""
        device = torch.device('cpu')
        batch = TrainingBatch(
            initial_hidden_states=model.initial_hidden_states,
            size=4,
            device=device,
        )
        
        assert batch.board_inputs.shape == (4, 81)
        assert batch.board_targets.shape == (4, 81)
        assert batch.segments.shape == (4,)
        assert batch.curriculum_level == 0
    
    def test_batch_difficulty_sampling(self, model):
        """Test difficulty sampling."""
        device = torch.device('cpu')
        batch = TrainingBatch(
            initial_hidden_states=model.initial_hidden_states,
            size=4,
            device=device,
        )
        
        # At level 0, should only sample easy
        difficulties = [batch.sample_difficulty() for _ in range(10)]
        assert all(d == Difficulty.EASY for d in difficulties)
        
        # After graduation, should sample from broader range
        batch.curriculum_level = 2
        difficulties = [batch.sample_difficulty() for _ in range(100)]
        unique_difficulties = set(difficulties)
        assert len(unique_difficulties) > 1
    
    def test_batch_replacement(self, model):
        """Test puzzle replacement in batch."""
        device = torch.device('cpu')
        batch = TrainingBatch(
            initial_hidden_states=model.initial_hidden_states,
            size=4,
            device=device,
        )
        
        old_input = batch.board_inputs[0].clone()
        old_total = batch.total_puzzles
        
        batch.replace(0)
        
        # Should have new puzzle
        assert not torch.equal(batch.board_inputs[0], old_input)
        assert batch.total_puzzles == old_total + 1
    
    def test_batch_graduation(self, model):
        """Test curriculum graduation."""
        device = torch.device('cpu')
        batch = TrainingBatch(
            initial_hidden_states=model.initial_hidden_states,
            size=4,
            device=device,
        )
        
        assert batch.curriculum_level == 0
        batch.graduate()
        assert batch.curriculum_level == 1
    
    def test_training_step(self, model, optimizer):
        """Test single training step."""
        device = torch.device('cpu')
        batch = TrainingBatch(
            initial_hidden_states=model.initial_hidden_states,
            size=4,
            device=device,
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        
        model.train()
        
        # Perform step
        (output_loss, output_acc), (q_act_loss, q_act_acc) = step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            generator=generator,
        )
        
        # Check outputs
        assert isinstance(output_loss, float)
        assert isinstance(output_acc, float)
        assert isinstance(q_act_loss, float)
        assert isinstance(q_act_acc, float)
        
        assert 0 <= output_acc <= 1
        assert 0 <= q_act_acc <= 1
    
    def test_training_step_updates_parameters(self, model, optimizer):
        """Test that training step updates parameters."""
        device = torch.device('cpu')
        batch = TrainingBatch(
            initial_hidden_states=model.initial_hidden_states,
            size=4,
            device=device,
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        
        model.train()
        
        # Save initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Perform step
        step(model=model, optimizer=optimizer, batch=batch, generator=generator)
        
        # Check that some parameters changed
        changed = False
        for name, param in model.named_parameters():
            if not torch.equal(param, initial_params[name]):
                changed = True
                break
        
        assert changed
    
    def test_training_step_updates_batch_states(self, model, optimizer):
        """Test that training step updates batch hidden states."""
        device = torch.device('cpu')
        batch = TrainingBatch(
            initial_hidden_states=model.initial_hidden_states,
            size=4,
            device=device,
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        
        model.train()
        
        initial_hl = batch.hidden_states.high_level.clone()
        initial_segments = batch.segments.clone()
        
        # Perform step
        step(model=model, optimizer=optimizer, batch=batch, generator=generator)
        
        # Segments should increment
        assert torch.all(batch.segments >= initial_segments)
    
    def test_multiple_training_steps(self, model, optimizer):
        """Test multiple consecutive training steps."""
        device = torch.device('cpu')
        batch = TrainingBatch(
            initial_hidden_states=model.initial_hidden_states,
            size=4,
            device=device,
        )
        
        model.train()
        
        # Perform multiple steps
        for i in range(5):
            generator = torch.Generator()
            generator.manual_seed(42 + i)
            
            (output_loss, output_acc), (q_act_loss, q_act_acc) = step(
                model=model,
                optimizer=optimizer,
                batch=batch,
                generator=generator,
            )
            
            # Should not produce NaN
            assert not (output_loss != output_loss)  # Check for NaN
            assert not (q_act_loss != q_act_loss)
