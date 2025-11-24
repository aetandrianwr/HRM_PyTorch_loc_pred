"""
Unit tests for Sudoku utilities.
"""

import pytest
from src.hrm.utils.sudoku import (
    Difficulty,
    generate_sudoku,
    sudoku_board_string,
    bit,
    box_index,
    first_empty_cell,
    fill_grid,
    solve,
)


class TestSudokuUtilities:
    """Test Sudoku utility functions."""
    
    def test_bit_function(self):
        """Test bit mask generation."""
        assert bit(1) == 1
        assert bit(2) == 2
        assert bit(3) == 4
        assert bit(9) == 256
    
    def test_box_index(self):
        """Test box index calculation."""
        assert box_index(0, 0) == 0
        assert box_index(0, 3) == 1
        assert box_index(0, 6) == 2
        assert box_index(3, 0) == 3
        assert box_index(4, 4) == 4
        assert box_index(8, 8) == 8
    
    def test_first_empty_cell(self):
        """Test finding first empty cell."""
        grid = [[1, 2, 3, 4, 5, 6, 7, 8, 9] for _ in range(9)]
        assert first_empty_cell(grid) is None
        
        grid[5][3] = 0
        assert first_empty_cell(grid) == (5, 3)
    
    def test_fill_grid(self):
        """Test grid filling."""
        grid = [[0] * 9 for _ in range(9)]
        result = fill_grid(grid)
        
        assert result is True
        
        # Check all cells are filled
        for row in grid:
            assert all(cell != 0 for cell in row)
            assert len(set(row)) == 9  # All different
    
    def test_solve_unique_solution(self):
        """Test solving puzzle with unique solution."""
        # Simple puzzle with unique solution
        puzzle = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ]
        
        test_grid = [row[:] for row in puzzle]
        num_solutions = solve(test_grid, limit=2)
        
        assert num_solutions == 1
    
    def test_generate_sudoku_very_easy(self):
        """Test generating very easy puzzle."""
        puzzle, solution = generate_sudoku(Difficulty.VERY_EASY)
        
        # Check dimensions
        assert len(puzzle) == 9
        assert len(solution) == 9
        assert all(len(row) == 9 for row in puzzle)
        assert all(len(row) == 9 for row in solution)
        
        # Count clues
        clues = sum(1 for row in puzzle for cell in row if cell != 0)
        assert 46 <= clues <= 50
        
        # Check solution is complete
        assert all(cell != 0 for row in solution for cell in row)
    
    def test_generate_sudoku_easy(self):
        """Test generating easy puzzle."""
        puzzle, solution = generate_sudoku(Difficulty.EASY)
        
        clues = sum(1 for row in puzzle for cell in row if cell != 0)
        assert 40 <= clues <= 45
    
    def test_generate_sudoku_medium(self):
        """Test generating medium puzzle."""
        puzzle, solution = generate_sudoku(Difficulty.MEDIUM)
        
        clues = sum(1 for row in puzzle for cell in row if cell != 0)
        assert 32 <= clues <= 39
    
    def test_generate_sudoku_hard(self):
        """Test generating hard puzzle."""
        puzzle, solution = generate_sudoku(Difficulty.HARD)
        
        clues = sum(1 for row in puzzle for cell in row if cell != 0)
        assert 28 <= clues <= 31
    
    def test_puzzle_has_unique_solution(self):
        """Test that generated puzzle has unique solution."""
        puzzle, solution = generate_sudoku(Difficulty.EASY)
        
        test_grid = [row[:] for row in puzzle]
        num_solutions = solve(test_grid, limit=2)
        
        assert num_solutions == 1
    
    def test_puzzle_clues_match_solution(self):
        """Test that puzzle clues are correct."""
        puzzle, solution = generate_sudoku(Difficulty.EASY)
        
        for i in range(9):
            for j in range(9):
                if puzzle[i][j] != 0:
                    assert puzzle[i][j] == solution[i][j]
    
    def test_sudoku_board_string(self):
        """Test board string formatting."""
        board = [[i + 1 if j < 5 else 0 for j in range(9)] for i in range(9)]
        string = sudoku_board_string(board)
        
        assert isinstance(string, str)
        assert '+-------+-------+-------+' in string
        assert '|' in string
        assert '.' in string  # For empty cells
