"""
Unit tests for RMSNorm.
"""

import pytest
import torch
from src.hrm.modeling.rmsnorm import rms_norm, RMSNorm


class TestRMSNorm:
    """Test RMS normalization."""
    
    def test_basic_normalization(self):
        """Test basic RMS normalization."""
        x = torch.randn(2, 3, 4)
        result = rms_norm(x)
        
        assert result.shape == x.shape
        assert result.dtype == x.dtype
    
    def test_dtype_preservation(self):
        """Test that output dtype matches input dtype."""
        x = torch.randn(2, 3, 4, dtype=torch.bfloat16)
        result = rms_norm(x)
        
        assert result.dtype == torch.bfloat16
    
    def test_variance_normalized(self):
        """Test that variance along last dimension is approximately 1."""
        x = torch.randn(10, 20, 30) * 5 + 10
        result = rms_norm(x)
        
        # Compute variance along last dimension
        variance = result.float().pow(2).mean(dim=-1)
        
        # Should be close to 1
        assert torch.allclose(variance, torch.ones_like(variance), rtol=0.1)
    
    def test_module_wrapper(self):
        """Test RMSNorm module wrapper."""
        module = RMSNorm(epsilon=1e-5)
        x = torch.randn(2, 3, 4)
        result = module(x)
        
        assert result.shape == x.shape
    
    def test_epsilon_effect(self):
        """Test that epsilon prevents division by zero."""
        x = torch.zeros(2, 3, 4)
        result = rms_norm(x, epsilon=1e-6)
        
        # Should not contain NaN or Inf
        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))
    
    def test_backward_pass(self):
        """Test that gradients flow correctly."""
        x = torch.randn(2, 3, 4, requires_grad=True)
        result = rms_norm(x)
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.any(torch.isnan(x.grad))
