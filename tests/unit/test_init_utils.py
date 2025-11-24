"""
Unit tests for initialization utilities.
"""

import pytest
import torch
import math
from src.hrm.modeling.init_utils import trunc_normal_init


class TestTruncNormalInit:
    """Test truncated normal initialization."""
    
    def test_basic_initialization(self):
        """Test basic tensor initialization."""
        shape = (10, 20)
        tensor = trunc_normal_init(shape)
        
        assert tensor.shape == shape
        assert tensor.dtype == torch.float32
    
    def test_custom_dtype(self):
        """Test initialization with custom dtype."""
        shape = (5, 5)
        tensor = trunc_normal_init(shape, dtype=torch.bfloat16)
        
        assert tensor.dtype == torch.bfloat16
    
    def test_zero_std(self):
        """Test initialization with zero std returns zeros."""
        shape = (3, 4)
        tensor = trunc_normal_init(shape, std=0.0)
        
        assert torch.all(tensor == 0)
    
    def test_within_bounds(self):
        """Test that values are within truncation bounds."""
        shape = (100, 100)
        std = 1.0
        lower = -2.0
        upper = 2.0
        
        tensor = trunc_normal_init(shape, std=std, lower=lower, upper=upper)
        
        # Values should be roughly within bounds * std (with some tolerance)
        assert tensor.min() >= lower * std - 0.3
        assert tensor.max() <= upper * std + 0.3
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        shape = (10, 10)
        
        gen1 = torch.Generator()
        gen1.manual_seed(42)
        tensor1 = trunc_normal_init(shape, generator=gen1)
        
        gen2 = torch.Generator()
        gen2.manual_seed(42)
        tensor2 = trunc_normal_init(shape, generator=gen2)
        
        assert torch.allclose(tensor1, tensor2)
    
    def test_different_seeds_differ(self):
        """Test that different seeds produce different results."""
        shape = (10, 10)
        
        gen1 = torch.Generator()
        gen1.manual_seed(42)
        tensor1 = trunc_normal_init(shape, generator=gen1)
        
        gen2 = torch.Generator()
        gen2.manual_seed(123)
        tensor2 = trunc_normal_init(shape, generator=gen2)
        
        assert not torch.allclose(tensor1, tensor2)
    
    def test_device_placement(self):
        """Test initialization on specific device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        shape = (5, 5)
        device = torch.device('cuda')
        tensor = trunc_normal_init(shape, device=device)
        
        assert tensor.device.type == 'cuda'
