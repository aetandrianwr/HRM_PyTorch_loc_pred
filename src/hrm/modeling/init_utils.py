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
    
    sqrt2 = math.sqrt(2.0)
    a = math.erf(lower / sqrt2)
    b = math.erf(upper / sqrt2)
    z = (b - a) / 2
    
    c = math.pow(2 * math.pi, -0.5)
    pdf_u = c * math.exp(-0.5 * math.pow(lower, 2))
    pdf_l = c * math.exp(-0.5 * math.pow(upper, 2))
    comp_std = std / math.sqrt(
        1.0 - (upper * pdf_u - lower * pdf_l) / z - math.pow((pdf_u - pdf_l) / z, 2)
    )
    
    # Generate uniform random values in [a, b]
    result = torch.empty(shape, dtype=dtype, device=device)
    if generator is not None:
        result.uniform_(a, b, generator=generator)
    else:
        result.uniform_(a, b)
    
    # Apply inverse error function
    result = torch.erfinv(result)
    
    # Scale and clip
    result = result * (sqrt2 * comp_std)
    result = torch.clamp(result, min=lower * comp_std, max=upper * comp_std)
    
    return result
