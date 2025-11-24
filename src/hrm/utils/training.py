"""
Training utilities and loss functions.
Faithful PyTorch conversion from Training.swift
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import random

from ..modeling.hrm import HRMACTInner, HiddenStates
from .sudoku import Difficulty, generate_sudoku


def sudoku_loss(
    model: HRMACTInner,
    hidden_states: HiddenStates,
    board_inputs: torch.Tensor,
    board_targets: torch.Tensor,
    segments: torch.Tensor,
    generator: torch.Generator = None,
) -> List[torch.Tensor]:
    """
    Compute loss for Sudoku task with ACT.
    
    Args:
        model: HRM-ACT model
        hidden_states: Current hidden states
        board_inputs: Input boards [batch_size, 81]
        board_targets: Target solutions [batch_size, 81]
        segments: Current segment count [batch_size]
        generator: Random generator
        
    Returns:
        List containing:
            [0] total_loss
            [1] output_loss
            [2] q_act_loss
            [3] is_halted (bool tensor)
            [4] output_accuracy
            [5] q_act_halt_accuracy
            [6] next_high_level_state
            [7] next_low_level_state
    """
    # Forward pass
    output = model(hidden_states=hidden_states, inputs=board_inputs)
    
    # Compute output loss
    output_logits = output.output
    output_loss = F.cross_entropy(
        output_logits.reshape(-1, output_logits.shape[-1]),
        board_targets.reshape(-1),
        reduction='none'
    ).reshape(board_targets.shape)
    
    # Mask: only compute loss for cells that were empty in input
    output_loss_mask = (board_inputs == 0).float()
    output_loss = output_loss * output_loss_mask
    
    # Compute output accuracy (all non-input cells must be correct)
    output_accuracy = (
        (output.output.argmax(dim=2) == board_targets) | (board_inputs != 0)
    ).min(dim=1)[0]
    
    # Q-ACT halt target: 1 if puzzle is solved, 0 otherwise
    q_act_halt_target = output_accuracy.long()
    
    # Compute next segment and check if should halt
    next_segments = segments + 1
    is_last_segment = next_segments > model.config.act.halt_max_steps
    is_halted = is_last_segment | (output.q_act_halt > output.q_act_continue)
    
    # Exploration: randomly set minimum halt segments
    device = board_inputs.device
    if generator is not None:
        halt_exploration = torch.rand(
            output.q_act_halt.shape, device=device, generator=generator
        ) < model.config.act.halt_exploration_probability
        
        min_halt_segments = torch.randint(
            2,
            model.config.act.halt_max_steps + 1,
            segments.shape,
            device=device,
            generator=generator,
        ) * halt_exploration.long()
    else:
        halt_exploration = torch.rand(output.q_act_halt.shape, device=device) < \
            model.config.act.halt_exploration_probability
        
        min_halt_segments = torch.randint(
            2,
            model.config.act.halt_max_steps + 1,
            segments.shape,
            device=device,
        ) * halt_exploration.long()
    
    is_halted = is_halted & (next_segments > min_halt_segments)
    
    # Get next segment output for Q-learning target
    with torch.no_grad():
        next_segment_output = model(hidden_states=output.hidden_states, inputs=board_inputs)
        next_q_act_halt = next_segment_output.q_act_halt
        next_q_act_continue = next_segment_output.q_act_continue
    
    # Compute Q-ACT continue target
    q_act_continue_target = torch.sigmoid(
        torch.where(
            is_last_segment,
            next_q_act_halt,
            torch.maximum(next_q_act_halt, next_q_act_continue),
        )
    )
    
    # Compute Q-ACT loss
    q_act_loss = (
        F.binary_cross_entropy_with_logits(
            output.q_act_halt, q_act_halt_target.float(), reduction='none'
        ) +
        F.binary_cross_entropy_with_logits(
            output.q_act_continue, q_act_continue_target, reduction='none'
        )
    ) / 2
    
    # Average losses
    avg_output_loss = output_loss.sum() / output_loss_mask.sum()
    avg_q_act_loss = q_act_loss.mean()
    
    # Compute accuracies
    avg_output_full_accuracy = (
        (output.output.argmax(dim=2) == board_targets) | (board_inputs != 0)
    ).float().mean()
    avg_q_act_halt_accuracy = ((output.q_act_halt >= 0) == output_accuracy).float().mean()
    
    return [
        avg_output_loss + avg_q_act_loss,  # total loss
        avg_output_loss,
        avg_q_act_loss,
        is_halted,
        avg_output_full_accuracy,
        avg_q_act_halt_accuracy,
        output.hidden_states.high_level,
        output.hidden_states.low_level,
    ]


class TrainingBatch:
    """
    Training batch with curriculum learning.
    """
    
    DIFFICULTIES = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD, Difficulty.EXTREME]
    
    CURRICULUM_DIFFICULTY_PROBAS = [
        [1.0, 0.0, 0.0, 0.0],  # stage 0: only easy
        [0.7, 0.3, 0.0, 0.0],  # stage 1: mostly easy, some medium
        [0.5, 0.4, 0.1, 0.0],  # stage 2: mix of easy, medium, some hard
        [0.3, 0.3, 0.3, 0.1],  # stage 3: mix of all difficulties
        [0.1, 0.3, 0.4, 0.2],  # stage 4: more hard and extreme
    ]
    
    def __init__(
        self,
        initial_hidden_states: HiddenStates,
        size: int,
        device: torch.device,
    ):
        self.initial_hidden_states = initial_hidden_states
        self.size = size
        self.device = device
        self.curriculum_level = 0
        self.total_puzzles = 0
        
        dtype = initial_hidden_states.high_level.dtype
        hidden_shape = initial_hidden_states.high_level.shape
        
        # Initialize batch tensors
        self.hidden_states = HiddenStates(
            high_level=torch.zeros(
                (size, 1, *hidden_shape), dtype=dtype, device=device
            ),
            low_level=torch.zeros(
                (size, 1, *hidden_shape), dtype=dtype, device=device
            ),
        )
        self.board_inputs = torch.zeros((size, 81), dtype=torch.long, device=device)
        self.board_targets = torch.zeros((size, 81), dtype=torch.long, device=device)
        self.segments = torch.zeros(size, dtype=torch.long, device=device)
        
        # Fill with initial puzzles
        for i in range(size):
            self.replace(i)
    
    def sample_difficulty(self) -> Difficulty:
        """Sample difficulty based on curriculum level."""
        probabilities = self.CURRICULUM_DIFFICULTY_PROBAS[self.curriculum_level]
        rand = random.random()
        cumulative = 0.0
        for idx, prob in enumerate(probabilities):
            cumulative += prob
            if rand < cumulative:
                return self.DIFFICULTIES[idx]
        return self.DIFFICULTIES[-1]
    
    def replace(self, idx: int):
        """Replace puzzle at index with new one."""
        # Reset hidden states
        self.hidden_states.high_level[idx] = self.initial_hidden_states.high_level.unsqueeze(0)
        self.hidden_states.low_level[idx] = self.initial_hidden_states.low_level.unsqueeze(0)
        self.segments[idx] = 0
        
        # Generate new puzzle
        puzzle, solution = generate_sudoku(self.sample_difficulty())
        
        # Flatten and convert to tensors
        puzzle_flat = [cell for row in puzzle for cell in row]
        solution_flat = [cell for row in solution for cell in row]
        
        self.board_inputs[idx] = torch.tensor(puzzle_flat, dtype=torch.long, device=self.device)
        self.board_targets[idx] = torch.tensor(solution_flat, dtype=torch.long, device=self.device)
        
        self.total_puzzles += 1
    
    def graduate(self):
        """Move to next curriculum level."""
        next_level = self.curriculum_level + 1
        if next_level >= len(self.CURRICULUM_DIFFICULTY_PROBAS):
            print("Reached highest curriculum level, cannot graduate.")
            return
        
        self.curriculum_level = next_level
        print(f"Graduated to level {self.curriculum_level}.")


def step(
    model: HRMACTInner,
    optimizer: torch.optim.Optimizer,
    batch: TrainingBatch,
    generator: torch.Generator,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Perform one training step.
    
    Args:
        model: HRM-ACT model
        optimizer: Optimizer
        batch: Training batch
        generator: Random generator
        
    Returns:
        ((output_loss, output_acc), (q_act_loss, q_act_acc))
    """
    optimizer.zero_grad()
    
    # Forward pass with loss computation
    losses = sudoku_loss(
        model=model,
        hidden_states=batch.hidden_states,
        board_inputs=batch.board_inputs,
        board_targets=batch.board_targets,
        segments=batch.segments,
        generator=generator,
    )
    
    total_loss = losses[0]
    output_loss = losses[1].item()
    q_act_loss = losses[2].item()
    is_halted = losses[3]
    output_acc = losses[4].item()
    q_act_acc = losses[5].item()
    next_hl_state = losses[6]
    next_ll_state = losses[7]
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    # Update batch states
    batch.hidden_states.high_level = next_hl_state.detach()
    batch.hidden_states.low_level = next_ll_state.detach()
    batch.segments += 1
    
    # Replace halted puzzles with new ones
    halted_indices = torch.where(is_halted)[0]
    for idx in halted_indices.cpu().tolist():
        batch.replace(idx)
    
    print(
        f"Output [{output_loss:.4f} {output_acc:.4f}] | "
        f"Q-ACT [{q_act_loss:.4f} {q_act_acc:.4f}] | "
        f"Puzzles [{batch.total_puzzles}] | "
        f"Curriculum Level [{batch.curriculum_level}]"
    )
    
    return ((output_loss, output_acc), (q_act_loss, q_act_acc))
