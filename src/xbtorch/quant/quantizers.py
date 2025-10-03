"""
Quantization utilities for fixed-point representation in XBTorch using QTorch.

Provides:
- Autograd-compatible fixed-point quantization functions
- Utility to compute numeric ranges based on fixed-point format

Modules
-------
FixedPointQuantF : torch.autograd.Function
    Custom autograd function wrapping QTorch's fixed-point quantizer.

Functions
---------
fixed_point_quantize_wrapper(input, wl, fl, symmetric=True, rounding="nearest")
    Convenience wrapper to quantize a tensor with given word length and fractional length.

get_range_fixed_point(num_format)
    Computes the representable range for a fixed-point number format.
"""

import torch
from qtorch.quant import fixed_point_quantize

class FixedPointQuantF(torch.autograd.Function):
    """
    Autograd-compatible fixed-point quantization function.

    Uses QTorch's fixed_point_quantize to quantize the input tensor to a fixed-point
    representation. The backward pass is a straight-through estimator (STE), returning
    the gradient unmodified.
    """

    @staticmethod
    def forward(ctx, input, wl, fl, symmetric=True, rounding='nearest'):
        """
        Forward pass: quantizes input to fixed-point.

        Parameters
        ----------
        input : torch.Tensor
            Tensor to quantize.
        wl : int
            Word length (total bits) for fixed-point representation.
        fl : int
            Fractional length (bits used for fraction).
        symmetric : bool, default=True
            Whether to use symmetric quantization.
        rounding : str, default='nearest'
            Rounding method (currently only nearest supported).

        Returns
        -------
        torch.Tensor
            Quantized tensor.
        """
        return fixed_point_quantize(input, wl=wl, fl=fl, symmetric=symmetric)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Straight-through estimator (STE) backward pass.

        Returns
        -------
        tuple
            Gradient for input and None for all other arguments.
        """
        return grad_output, None, None, None, None

def fixed_point_quantize_wrapper(input, wl, fl, symmetric=True, rounding="nearest"):
    """
    Convenience wrapper for FixedPointQuantF.

    Parameters
    ----------
    input : torch.Tensor
        Tensor to quantize.
    wl : int
        Word length (total bits) for fixed-point representation.
    fl : int
        Fractional length (bits used for fraction).
    symmetric : bool, default=True
        Whether to use symmetric quantization.
    rounding : str, default='nearest'
        Rounding method.

    Returns
    -------
    torch.Tensor
        Quantized tensor.
    """
    return FixedPointQuantF.apply(input, wl, fl, symmetric, rounding)

def get_range_fixed_point(num_format):
    """
    Compute the numeric range for a given fixed-point format.

    Parameters
    ----------
    num_format : object
        Object with attributes:
        - wl : word length (bits)
        - fl : fractional length (bits)
        - symmetric : bool, symmetric vs asymmetric quantization

    Returns
    -------
    tuple
        (min_value, max_value) representable by the fixed-point format.
    """
    if (num_format.symmetric):
        return (-(2**(num_format.wl-num_format.fl-1) - 2**(-num_format.fl)), 2**(num_format.wl-num_format.fl-1) - 2**(-num_format.fl))
    else:
        return (-2**(num_format.wl-num_format.fl-1), 2**(num_format.wl-num_format.fl-1) - 2**(-num_format.fl))