"""
Sudoku puzzle generation and solving utilities.
Faithful PyTorch conversion from Sudoku.swift
"""

import random
from typing import List, Tuple, Optional
from enum import Enum


class Difficulty(Enum):
    """Sudoku difficulty levels."""
    VERY_EASY = "very-easy"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"


def bit(value: int) -> int:
    """Get bit mask for a value (1-9)."""
    return 1 << (value - 1)


def box_index(row: int, col: int) -> int:
    """Get 3x3 box index for a cell."""
    return (row // 3) * 3 + (col // 3)


def build_masks(grid: List[List[int]]) -> Tuple[List[int], List[int], List[int]]:
    """
    Build bit masks for rows, columns, and boxes.
    
    Args:
        grid: 9x9 Sudoku grid
        
    Returns:
        Tuple of (rows, cols, boxes) bit masks
    """
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9
    
    for r in range(9):
        row_mask = 0
        for c in range(9):
            v = grid[r][c]
            if v != 0:
                b = bit(v)
                row_mask |= b
                cols[c] |= b
                boxes[box_index(r, c)] |= b
        rows[r] = row_mask
    
    return rows, cols, boxes


def first_empty_cell(grid: List[List[int]]) -> Optional[Tuple[int, int]]:
    """Find first empty cell in grid."""
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                return (r, c)
    return None


def fill_grid_rec(
    grid: List[List[int]],
    rows: List[int],
    cols: List[int],
    boxes: List[int],
) -> bool:
    """
    Recursively fill grid with valid numbers.
    
    Args:
        grid: 9x9 grid to fill
        rows: Row bit masks
        cols: Column bit masks
        boxes: Box bit masks
        
    Returns:
        True if successfully filled
    """
    cell = first_empty_cell(grid)
    if cell is None:
        return True
    
    row, col = cell
    numbers = list(range(1, 10))
    random.shuffle(numbers)
    
    b_idx = box_index(row, col)
    used = rows[row] | cols[col] | boxes[b_idx]
    
    for num in numbers:
        b = bit(num)
        if (used & b) != 0:
            continue
        
        grid[row][col] = num
        rows[row] |= b
        cols[col] |= b
        boxes[b_idx] |= b
        
        if fill_grid_rec(grid, rows, cols, boxes):
            return True
        
        grid[row][col] = 0
        rows[row] &= ~b
        cols[col] &= ~b
        boxes[b_idx] &= ~b
    
    return False


def fill_grid(grid: List[List[int]]) -> bool:
    """Fill a Sudoku grid with valid solution."""
    rows, cols, boxes = build_masks(grid)
    return fill_grid_rec(grid, rows, cols, boxes)


def solve_rec(
    grid: List[List[int]],
    solutions: List[int],
    limit: int,
    rows: List[int],
    cols: List[int],
    boxes: List[int],
) -> bool:
    """
    Recursively solve grid and count solutions.
    
    Args:
        grid: 9x9 grid to solve
        solutions: List with solution count (mutable)
        limit: Maximum solutions to find
        rows: Row bit masks
        cols: Column bit masks
        boxes: Box bit masks
        
    Returns:
        True if limit reached
    """
    if solutions[0] >= limit:
        return True
    
    cell = first_empty_cell(grid)
    if cell is None:
        solutions[0] += 1
        return solutions[0] >= limit
    
    row, col = cell
    b_idx = box_index(row, col)
    used = rows[row] | cols[col] | boxes[b_idx]
    
    for num in range(1, 10):
        b = bit(num)
        if (used & b) != 0:
            continue
        
        grid[row][col] = num
        rows[row] |= b
        cols[col] |= b
        boxes[b_idx] |= b
        
        if solve_rec(grid, solutions, limit, rows, cols, boxes):
            return True
        
        grid[row][col] = 0
        rows[row] &= ~b
        cols[col] &= ~b
        boxes[b_idx] &= ~b
    
    return False


def solve(grid: List[List[int]], limit: int = 2) -> int:
    """
    Count solutions for a puzzle (up to limit).
    
    Args:
        grid: Puzzle to solve (modified in place)
        limit: Maximum solutions to count
        
    Returns:
        Number of solutions found (up to limit)
    """
    solutions = [0]
    rows, cols, boxes = build_masks(grid)
    solve_rec(grid, solutions, limit, rows, cols, boxes)
    return solutions[0]


def generate_sudoku(difficulty: Difficulty) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Generate a Sudoku puzzle with solution.
    
    Args:
        difficulty: Desired difficulty level
        
    Returns:
        Tuple of (puzzle, solution) as 9x9 grids
    """
    # Create empty board and fill it
    board = [[0] * 9 for _ in range(9)]
    fill_grid(board)
    solution = [row[:] for row in board]
    
    # Determine target clue range based on difficulty
    target_clues = {
        Difficulty.VERY_EASY: range(46, 51),
        Difficulty.EASY: range(40, 46),
        Difficulty.MEDIUM: range(32, 40),
        Difficulty.HARD: range(28, 32),
        Difficulty.EXTREME: range(17, 28),
    }[difficulty]
    
    # Create puzzle by removing cells
    puzzle = [row[:] for row in board]
    cells = list(range(81))
    random.shuffle(cells)
    
    cursor = 0
    clues = 81
    
    # Remove cells while maintaining unique solution
    while cursor < len(cells) and clues > max(target_clues):
        idx = cells[cursor]
        cursor += 1
        r = idx // 9
        c = idx % 9
        backup = puzzle[r][c]
        puzzle[r][c] = 0
        
        # Check if puzzle still has unique solution
        test = [row[:] for row in puzzle]
        solution_count = solve(test, limit=2)
        
        if solution_count != 1:
            puzzle[r][c] = backup
        else:
            clues -= 1
    
    # Try to reach lower bound of target range
    if clues > min(target_clues):
        for j in range(cursor, len(cells)):
            if clues <= min(target_clues):
                break
            idx = cells[j]
            r = idx // 9
            c = idx % 9
            backup = puzzle[r][c]
            puzzle[r][c] = 0
            
            test = [row[:] for row in puzzle]
            solution_count = solve(test, limit=2)
            
            if solution_count != 1:
                puzzle[r][c] = backup
            else:
                clues -= 1
    
    return puzzle, solution


def sudoku_board_string(board: List[List[int]]) -> str:
    """
    Convert Sudoku board to string representation.
    
    Args:
        board: 9x9 Sudoku grid
        
    Returns:
        Formatted string representation
    """
    horizontal_line = "+-------+-------+-------+"
    result = [horizontal_line]
    
    for row_idx, row in enumerate(board):
        line = "|"
        for col_idx, cell in enumerate(row):
            display_value = "." if cell == 0 else str(cell)
            line += f" {display_value}"
            if (col_idx + 1) % 3 == 0:
                line += " |"
        result.append(line)
        
        if (row_idx + 1) % 3 == 0:
            result.append(horizontal_line)
    
    return "\n".join(result)
