"""
Truncated normal initialization utilities.
Faithful PyTorch conversion from TruncNormalInit.swift
"""

import torch
import math


def trunc_normal_init(
    shape: tuple,
    std: float = 1.0,
    lower: float = -2.0,
    upper: float = 2.0,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator = None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Initialize tensor with truncated normal distribution.
    
    Args:
        shape: Shape of output tensor
        std: Standard deviation
        lower: Lower bound in standard deviations
        upper: Upper bound in standard deviations
        dtype: Data type
        generator: Random generator for reproducibility
        device: Device to create tensor on
        
    Returns:
        Initialized tensor
    """
    if std == 0.0:
        return torch.zeros(shape, dtype=dtype, device=device)
    
    # Use PyTorch's built-in trunc_normal_ for stability
    # Convert to CPU float32 for initialization, then convert back
    if dtype == torch.bfloat16 or device.type == 'cuda':
        # Initialize on CPU with float32 for numerical stability
        result = torch.empty(shape, dtype=torch.float32, device='cpu')
        torch.nn.init.trunc_normal_(result, mean=0.0, std=std, a=lower*std, b=upper*std)
        # Move to target device and dtype
        result = result.to(device=device, dtype=dtype)
    else:
        result = torch.empty(shape, dtype=dtype, device=device)
        torch.nn.init.trunc_normal_(result, mean=0.0, std=std, a=lower*std, b=upper*std)
    
    return result
