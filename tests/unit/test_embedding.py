"""
Unit tests for Embedding layer.
"""

import pytest
import torch
from src.hrm.modeling.embedding import Embedding


class TestEmbedding:
    """Test custom Embedding layer."""
    
    def test_basic_forward(self):
        """Test basic embedding lookup."""
        layer = Embedding(vocab_size=100, dim=64, init_std=0.02)
        x = torch.randint(0, 100, (2, 10))
        output = layer(x)
        
        assert output.shape == (2, 10, 64)
    
    def test_embedding_shape(self):
        """Test embedding parameter shape."""
        layer = Embedding(vocab_size=50, dim=32, init_std=0.02)
        
        assert layer.embeddings.shape == (50, 32)
    
    def test_correct_lookup(self):
        """Test that correct embeddings are retrieved."""
        layer = Embedding(vocab_size=10, dim=4, init_std=0.02)
        
        # Manually set embeddings for testing
        layer.embeddings.data = torch.arange(40, dtype=torch.float32).reshape(10, 4)
        
        indices = torch.tensor([0, 5, 9])
        output = layer(indices)
        
        assert torch.allclose(output[0], layer.embeddings[0])
        assert torch.allclose(output[1], layer.embeddings[5])
        assert torch.allclose(output[2], layer.embeddings[9])
    
    def test_batch_lookup(self):
        """Test batched embedding lookup."""
        layer = Embedding(vocab_size=100, dim=32, init_std=0.02)
        x = torch.randint(0, 100, (4, 8, 16))
        output = layer(x)
        
        assert output.shape == (4, 8, 16, 32)
    
    def test_initialization_std(self):
        """Test that embeddings are initialized with correct std."""
        init_std = 0.01
        layer = Embedding(vocab_size=1000, dim=128, init_std=init_std)
        
        std = layer.embeddings.std().item()
        
        # Should be roughly the requested std
        assert abs(std - init_std) < 0.01
    
    def test_reproducibility(self):
        """Test reproducible initialization."""
        gen1 = torch.Generator()
        gen1.manual_seed(42)
        layer1 = Embedding(vocab_size=100, dim=64, init_std=0.02, generator=gen1)
        
        gen2 = torch.Generator()
        gen2.manual_seed(42)
        layer2 = Embedding(vocab_size=100, dim=64, init_std=0.02, generator=gen2)
        
        assert torch.allclose(layer1.embeddings, layer2.embeddings)
    
    def test_dtype(self):
        """Test custom dtype."""
        layer = Embedding(vocab_size=50, dim=32, init_std=0.02, dtype=torch.bfloat16)
        
        assert layer.embeddings.dtype == torch.bfloat16
    
    def test_backward_pass(self):
        """Test gradient computation."""
        layer = Embedding(vocab_size=50, dim=32, init_std=0.02)
        x = torch.randint(0, 50, (2, 10))
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        assert layer.embeddings.grad is not None
    
    def test_device_placement(self):
        """Test initialization on specific device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        layer = Embedding(vocab_size=50, dim=32, init_std=0.02, device=device)
        
        assert layer.embeddings.device.type == 'cuda'
