"""
Unit tests for Linear layer.
"""

import pytest
import torch
from src.hrm.modeling.linear import Linear


class TestLinear:
    """Test custom Linear layer."""
    
    def test_basic_forward(self):
        """Test basic forward pass."""
        layer = Linear(10, 20)
        x = torch.randn(2, 3, 10)
        output = layer(x)
        
        assert output.shape == (2, 3, 20)
    
    def test_no_bias(self):
        """Test linear layer without bias."""
        layer = Linear(10, 20, bias=False)
        
        assert layer.bias is None
        
        x = torch.randn(2, 10)
        output = layer(x)
        
        assert output.shape == (2, 20)
    
    def test_with_bias(self):
        """Test linear layer with bias."""
        layer = Linear(10, 20, bias=True)
        
        assert layer.bias is not None
        assert layer.bias.shape == (20,)
    
    def test_weight_initialization(self):
        """Test that weights are initialized with truncated normal."""
        layer = Linear(100, 100)
        
        # Weights should have reasonable std
        std = layer.weight.std().item()
        expected_std = 1.0 / (100 ** 0.5)
        
        # Should be roughly in the right range
        assert abs(std - expected_std) < 0.05
    
    def test_reproducibility(self):
        """Test reproducible initialization."""
        gen1 = torch.Generator()
        gen1.manual_seed(42)
        layer1 = Linear(10, 20, generator=gen1)
        
        gen2 = torch.Generator()
        gen2.manual_seed(42)
        layer2 = Linear(10, 20, generator=gen2)
        
        assert torch.allclose(layer1.weight, layer2.weight)
        assert torch.allclose(layer1.bias, layer2.bias)
    
    def test_dtype(self):
        """Test custom dtype."""
        layer = Linear(10, 20, dtype=torch.bfloat16)
        
        assert layer.weight.dtype == torch.bfloat16
        assert layer.bias.dtype == torch.bfloat16
    
    def test_backward_pass(self):
        """Test gradient computation."""
        layer = Linear(10, 20)
        x = torch.randn(2, 10, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None
    
    def test_device_placement(self):
        """Test initialization on specific device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        layer = Linear(10, 20, device=device)
        
        assert layer.weight.device.type == 'cuda'
        assert layer.bias.device.type == 'cuda'
