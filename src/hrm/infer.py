"""
Inference script for HRM-ACT model.
Faithful PyTorch conversion from HierarchicalReasoningModel.swift (infer function)
"""

import torch
import sys
import argparse

from .modeling import (
    HRMACTInner,
    HRMACTModelConfig,
    TransformerConfig,
    ACTConfig,
    HiddenStates,
)
from .utils import Difficulty, generate_sudoku, sudoku_board_string


def sigmoid(x):
    """Sigmoid function."""
    return torch.sigmoid(x)


def infer(checkpoint_path: str, difficulty: Difficulty):
    """
    Run inference on a random puzzle.
    
    Args:
        checkpoint_path: Path to model checkpoint
        difficulty: Puzzle difficulty
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Set random seed
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed(42)
    
    # Create generator
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    
    # Create model configuration (must match training)
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
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("Loaded model!\n")
    
    # Generate puzzle
    raw_puzzle, raw_solution = generate_sudoku(difficulty)
    
    print("Puzzle:")
    print(sudoku_board_string(raw_puzzle))
    print("\nSolution:")
    print(sudoku_board_string(raw_solution))
    
    # Prepare input
    puzzle_flat = [cell for row in raw_puzzle for cell in row]
    solution_flat = [cell for row in raw_solution for cell in row]
    
    puzzle_in = torch.tensor([puzzle_flat], dtype=torch.long, device=device)
    
    # Initialize hidden states
    hidden_states = HiddenStates(
        high_level=model.initial_hidden_states.high_level.unsqueeze(0).unsqueeze(0),
        low_level=model.initial_hidden_states.low_level.unsqueeze(0).unsqueeze(0),
    )
    
    # Run inference segments
    with torch.no_grad():
        for segment in range(1, config.act.halt_max_steps + 1):
            print(f"\nSegment {segment}")
            
            # Forward pass
            output = model(hidden_states=hidden_states, inputs=puzzle_in)
            hidden_states = output.hidden_states
            
            # Get predictions
            predictions = output.output[0].argmax(dim=1).cpu().tolist()
            
            # Compute accuracy
            accurate_squares = 0
            predicted_squares = 0
            predicted_flat_board = []
            
            for puzzle_sq, solution_sq, pred_sq in zip(puzzle_flat, solution_flat, predictions):
                if puzzle_sq != 0:
                    predicted_flat_board.append(puzzle_sq)
                else:
                    if pred_sq == solution_sq:
                        accurate_squares += 1
                    predicted_squares += 1
                    predicted_flat_board.append(pred_sq)
            
            # Convert to 2D board
            predicted_board = []
            for i in range(0, 81, 9):
                predicted_board.append(predicted_flat_board[i:i+9])
            
            print(f"Predicted solution ({accurate_squares} / {predicted_squares}):")
            print(sudoku_board_string(predicted_board))
            
            # Check Q-ACT values
            q_halt = sigmoid(output.q_act_halt[0]).item()
            q_continue = sigmoid(output.q_act_continue[0]).item()
            print(f"Q (halt - continue): {q_halt:.4f} - {q_continue:.4f}")
            
            # Check if should halt
            if q_halt > q_continue:
                print("Halting.")
                break


def main():
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(
        description='Run inference with HRM-ACT model on Sudoku puzzles'
    )
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to model checkpoint (.pt file)'
    )
    parser.add_argument(
        'difficulty',
        type=str,
        choices=['very-easy', 'easy', 'medium', 'hard', 'extreme'],
        help='Puzzle difficulty level'
    )
    
    args = parser.parse_args()
    
    # Map difficulty string to enum
    difficulty_map = {
        'very-easy': Difficulty.VERY_EASY,
        'easy': Difficulty.EASY,
        'medium': Difficulty.MEDIUM,
        'hard': Difficulty.HARD,
        'extreme': Difficulty.EXTREME,
    }
    
    difficulty = difficulty_map[args.difficulty]
    
    infer(args.checkpoint, difficulty)


if __name__ == '__main__':
    main()
