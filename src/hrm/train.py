"""
Training script for HRM-ACT model.
Faithful PyTorch conversion from HierarchicalReasoningModel.swift (train function)
"""

import torch
import os
from safetensors.torch import save_file

from .modeling import (
    HRMACTInner,
    HRMACTModelConfig,
    TransformerConfig,
    ACTConfig,
)
from .utils import TrainingBatch, step


def train():
    """Train HRM-ACT model on Sudoku puzzles."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Set random seed
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed(42)
    
    # Create random generator
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    
    # Create model configuration
    config = HRMACTModelConfig(
        seq_len=9 * 9,
        vocab_size=10,
        high_level_cycles=2,
        low_level_cycles=2,
        transformers=TransformerConfig(
            num_layers=4,
            hidden_size=256,
            num_heads=4,
            expansion=4.0,
        ),
        act=ACTConfig(
            halt_max_steps=16,
            halt_exploration_probability=0.1,
        ),
    )
    
    # Create model
    print("Initializing model...")
    model = HRMACTInner(config=config, generator=generator, device=device)
    model = model.to(device)
    model.train()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    
    # Create training batch
    print("Creating training batch...")
    batch = TrainingBatch(
        initial_hidden_states=model.initial_hidden_states,
        size=512,
        device=device,
    )
    
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    step_idx = 0
    steps_since_graduation = 0
    accuracy_history = [0.0] * 300
    
    print("Starting training...\n")
    
    while True:
        step_idx += 1
        steps_since_graduation += 1
        print(f"Step {step_idx}")
        
        # Create step generator
        step_gen = torch.Generator(device=device)
        step_gen.manual_seed(generator.initial_seed() + step_idx)
        
        # Perform training step
        (output_loss, output_acc), (q_act_loss, q_act_acc) = step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            generator=step_gen,
        )
        
        # Save checkpoint
        if step_idx == 1 or step_idx % 250 == 0:
            checkpoint_path = f'checkpoints/checkpoint-{step_idx}.pt'
            print(f"Saving checkpoint to {checkpoint_path}")
            torch.save({
                'step': step_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, checkpoint_path)
            
            # Also save in safetensors format
            safetensors_path = f'checkpoints/checkpoint-{step_idx}.safetensors'
            save_file(model.state_dict(), safetensors_path)
        
        # Update accuracy history
        accuracy_history.pop(0)
        accuracy_history.append(output_acc)
        avg_rolling_accuracy = sum(accuracy_history) / len(accuracy_history)
        
        # Curriculum learning graduation
        if avg_rolling_accuracy >= 0.85 and steps_since_graduation >= 300:
            steps_since_graduation = 0
            batch.graduate()
        
        print()


if __name__ == '__main__':
    train()
