import torch
from qtorch.quant import fixed_point_quantize

class FixedPointQuantF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, wl, fl, symmetric=True, rounding='nearest'):
        return fixed_point_quantize(input, wl=wl, fl=fl, symmetric=symmetric)

    @staticmethod
    def backward(ctx, grad_output):
        # matching ctor args, todo: find a better way to implement this
        return grad_output, None, None, None, None

def fixed_point_quantize_wrapper(input, wl, fl, symmetric=True, rounding="nearest"):
    return FixedPointQuantF.apply(input, wl, fl, symmetric, rounding)

def get_range_fixed_point(num_format):
    if (num_format.symmetric):
        return (-(2**(num_format.wl-num_format.fl-1) - 2**(-num_format.fl)), 2**(num_format.wl-num_format.fl-1) - 2**(-num_format.fl))
    else:
        return (-2**(num_format.wl-num_format.fl-1), 2**(num_format.wl-num_format.fl-1) - 2**(-num_format.fl))