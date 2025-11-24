"""
End-to-end tests for the complete system.
"""

import pytest
import torch
import os
import tempfile
from src.hrm.modeling import (
    HRMACTInner,
    HRMACTModelConfig,
    TransformerConfig,
    ACTConfig,
    HiddenStates,
)
from src.hrm.utils import (
    Difficulty,
    generate_sudoku,
    TrainingBatch,
    step,
)


class TestEndToEnd:
    """End-to-end system tests."""
    
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
                halt_max_steps=8,
                halt_exploration_probability=0.1,
            ),
            dtype=torch.float32,
        )
    
    def test_full_training_loop(self, config):
        """Test complete training loop for multiple steps."""
        device = torch.device('cpu')
        
        # Create model
        generator = torch.Generator()
        generator.manual_seed(42)
        model = HRMACTInner(config=config, generator=generator, device=device)
        model.train()
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
        
        # Create batch
        batch = TrainingBatch(
            initial_hidden_states=model.initial_hidden_states,
            size=8,
            device=device,
        )
        
        # Train for a few steps
        for step_idx in range(5):
            gen = torch.Generator()
            gen.manual_seed(42 + step_idx)
            
            (output_loss, output_acc), (q_act_loss, q_act_acc) = step(
                model=model,
                optimizer=optimizer,
                batch=batch,
                generator=gen,
            )
            
            # Verify metrics are valid
            assert isinstance(output_loss, float)
            assert isinstance(output_acc, float)
            assert 0 <= output_acc <= 1
    
    def test_save_and_load_checkpoint(self, config):
        """Test saving and loading model checkpoint."""
        device = torch.device('cpu')
        
        # Create and train model
        generator = torch.Generator()
        generator.manual_seed(42)
        model1 = HRMACTInner(config=config, generator=generator, device=device)
        
        optimizer = torch.optim.AdamW(model1.parameters(), lr=1e-4)
        
        batch = TrainingBatch(
            initial_hidden_states=model1.initial_hidden_states,
            size=4,
            device=device,
        )
        
        # Train for one step
        gen = torch.Generator()
        gen.manual_seed(42)
        step(model=model1, optimizer=optimizer, batch=batch, generator=gen)
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            torch.save({
                'model_state_dict': model1.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, checkpoint_path)
            
            # Create new model and load
            generator2 = torch.Generator()
            generator2.manual_seed(99)  # Different seed
            model2 = HRMACTInner(config=config, generator=generator2, device=device)
            
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model2.load_state_dict(checkpoint['model_state_dict'])
            
            # Compare parameters
            for (name1, param1), (name2, param2) in zip(
                model1.named_parameters(), model2.named_parameters()
            ):
                assert name1 == name2
                assert torch.allclose(param1, param2)
        
        finally:
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)
    
    def test_inference_on_puzzle(self, config):
        """Test inference on a Sudoku puzzle."""
        device = torch.device('cpu')
        
        # Create model
        generator = torch.Generator()
        generator.manual_seed(42)
        model = HRMACTInner(config=config, generator=generator, device=device)
        model.eval()
        
        # Generate puzzle
        puzzle, solution = generate_sudoku(Difficulty.EASY)
        
        # Prepare input
        puzzle_flat = [cell for row in puzzle for cell in row]
        solution_flat = [cell for row in solution for cell in row]
        puzzle_tensor = torch.tensor([puzzle_flat], dtype=torch.long, device=device)
        
        # Initialize hidden states
        hidden_states = HiddenStates(
            high_level=model.initial_hidden_states.high_level.unsqueeze(0).unsqueeze(0),
            low_level=model.initial_hidden_states.low_level.unsqueeze(0).unsqueeze(0),
        )
        
        # Run inference
        with torch.no_grad():
            for segment in range(config.act.halt_max_steps):
                output = model(hidden_states=hidden_states, inputs=puzzle_tensor)
                hidden_states = output.hidden_states
                
                # Check output shapes
                assert output.output.shape == (1, 81, 10)
                assert output.q_act_halt.shape == (1,)
                assert output.q_act_continue.shape == (1,)
                
                # Get predictions
                predictions = output.output[0].argmax(dim=1)
                
                # Check if should halt
                if output.q_act_halt[0] > output.q_act_continue[0]:
                    break
    
    def test_curriculum_learning_progression(self, config):
        """Test that curriculum learning progresses through levels."""
        device = torch.device('cpu')
        
        # Create model
        generator = torch.Generator()
        generator.manual_seed(42)
        model = HRMACTInner(config=config, generator=generator, device=device)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create batch
        batch = TrainingBatch(
            initial_hidden_states=model.initial_hidden_states,
            size=8,
            device=device,
        )
        
        assert batch.curriculum_level == 0
        
        # Simulate high accuracy for graduation
        batch.graduate()
        assert batch.curriculum_level == 1
        
        batch.graduate()
        assert batch.curriculum_level == 2
    
    def test_halt_mechanism(self, config):
        """Test that ACT halting mechanism works."""
        device = torch.device('cpu')
        
        # Create model
        generator = torch.Generator()
        generator.manual_seed(42)
        model = HRMACTInner(config=config, generator=generator, device=device)
        model.eval()
        
        # Generate puzzle
        puzzle, _ = generate_sudoku(Difficulty.EASY)
        puzzle_flat = [cell for row in puzzle for cell in row]
        puzzle_tensor = torch.tensor([puzzle_flat], dtype=torch.long, device=device)
        
        # Initialize hidden states
        hidden_states = HiddenStates(
            high_level=model.initial_hidden_states.high_level.unsqueeze(0).unsqueeze(0),
            low_level=model.initial_hidden_states.low_level.unsqueeze(0).unsqueeze(0),
        )
        
        # Run until halt or max steps
        halted = False
        with torch.no_grad():
            for segment in range(config.act.halt_max_steps):
                output = model(hidden_states=hidden_states, inputs=puzzle_tensor)
                hidden_states = output.hidden_states
                
                # Check Q values are valid
                assert not torch.isnan(output.q_act_halt)
                assert not torch.isnan(output.q_act_continue)
                
                if output.q_act_halt[0] > output.q_act_continue[0]:
                    halted = True
                    break
        
        # Should either halt or reach max steps
        assert halted or segment == config.act.halt_max_steps - 1
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_training(self, config):
        """Test training on GPU."""
        device = torch.device('cuda')
        
        # Create model
        generator = torch.Generator(device=device)
        generator.manual_seed(42)
        model = HRMACTInner(config=config, generator=generator, device=device)
        model = model.to(device)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create batch
        batch = TrainingBatch(
            initial_hidden_states=model.initial_hidden_states,
            size=4,
            device=device,
        )
        
        # Perform training step
        gen = torch.Generator(device=device)
        gen.manual_seed(42)
        
        (output_loss, output_acc), (q_act_loss, q_act_acc) = step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            generator=gen,
        )
        
        # Should complete without error
        assert isinstance(output_loss, float)
