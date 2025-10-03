"""
WAGE quantization routines adapted from QPyTorch example:
https://github.com/Tiiiger/QPyTorch/blob/master/examples/WAGE/wage_qtorch.py

This module implements fixed-point quantization for weights, activations,
and gradients, as well as a WAGEQuantizer wrapper for PyTorch modules.

Functions
---------
shift(x)
    Scales tensor x to prevent overflow during fixed-point quantization.
C(x, bits)
    Clamps tensor x to the representable range given a bit width.
QW(x, bits, scale=1.0, mode="nearest")
    Quantizes weights x using fixed-point representation with scaling.
QG(x, bits_G, lr, mode="nearest")
    Quantizes gradients x to a fixed-point representation for WAGE updates.

Classes
-------
WAGEQuantizer
    PyTorch module that performs activation and error quantization using WAGE.
"""

import torch
from torch.nn import Module
from qtorch.quant import fixed_point_quantize, quantizer
from qtorch import FixedPoint


def shift(x):
    """
    Scales tensor to prevent overflow in fixed-point quantization.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Scaled tensor.
    """
    max_entry = x.abs().max()
    return x / 2.0 ** torch.ceil(torch.log2(max_entry))

def C(x, bits):
    """
    Clamps the input to a representable range given the number of bits.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    bits : int
        Number of quantization bits.

    Returns
    -------
    torch.Tensor
        Clamped tensor.
    """
    if bits > 15 or bits == 1:
        delta = 0
    else:
        delta = 1.0 / (2.0 ** (bits - 1))
    upper = 1 - delta
    lower = -1 + delta
    return torch.clamp(x, lower, upper)


def QW(x, bits, scale=1.0, mode="nearest"):
    """
    Quantizes weights x to fixed-point representation.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    bits : int
        Bit-width of weight representation.
    scale : float, optional
        Scaling factor to normalize the layer weights.
    mode : str, optional
        Rounding mode: "nearest" or "stochastic".

    Returns
    -------
    torch.Tensor
        Quantized weights.
    """
    y = fixed_point_quantize(
        x, wl=bits, fl=bits - 1, clamp=True, symmetric=True, rounding=mode
    )
    # per layer scaling
    if scale > 1.8:
        y /= scale
    return y

def QG(x, bits_G, lr, mode="nearest"):
    """
    Quantizes gradients x for WAGE updates.

    Parameters
    ----------
    x : torch.Tensor
        Gradient tensor.
    bits_G : int
        Bit-width of gradient representation.
    lr : float
        Learning rate scaling factor.
    mode : str, optional
        Rounding mode: "nearest" or "stochastic".

    Returns
    -------
    torch.Tensor
        Quantized gradients.
    """
    x = shift(x)
    lr = lr / (2.0 ** (bits_G - 1))
    norm = fixed_point_quantize(
        lr * x, wl=bits_G, fl=bits_G - 1, clamp=False, symmetric=True, rounding=mode
    )
    return norm

class WAGEQuantizer(Module):
    """
    PyTorch module that performs WAGE quantization for activations and
    errors using QPyTorch's quantizer.

    Parameters
    ----------
    bits_A : int
        Bit-width for activation quantization.
    bits_E : int
        Bit-width for error/gradient quantization.
    A_mode : str, optional
        Rounding mode for activations.
    E_mode : str, optional
        Rounding mode for errors.
    """
    def __init__(self, bits_A, bits_E, A_mode="nearest", E_mode="nearest"):
        super(WAGEQuantizer, self).__init__()
        self.activate_number = (
            FixedPoint(wl=bits_A, fl=bits_A - 1, clamp=True, symmetric=True)
            if bits_A != -1
            else None
        )
        self.error_number = (
            FixedPoint(wl=bits_E, fl=bits_E - 1, clamp=True, symmetric=True)
            if bits_E != -1
            else None
        )
        self.quantizer = quantizer(
            forward_number=self.activate_number,
            forward_rounding=A_mode,
            backward_number=self.error_number,
            backward_rounding=E_mode,
            clamping_grad_zero=True,
            backward_hooks=[shift],
        )

    def forward(self, x):
        return self.quantizer(x)
