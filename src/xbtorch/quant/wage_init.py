"""
Initialization routines for WAGE quantized networks, adapted from QPyTorch:
https://github.com/Tiiiger/QPyTorch/blob/master/examples/WAGE/wage_qtorch.py

Functions
---------
truncated_normal_(tensor, mean=0, std=1)
    Fill a tensor with values drawn from a truncated normal distribution.
scale_limit(param, limit, bits_W)
    Compute a weight scaling factor based on the WAGE quantization bit width.
wage_init_(tensor, bits_W, factor=2.0, mode="fan_in")
    Initialize a tensor for WAGE quantization with appropriate scaling limits.
"""

import math
import numpy as np

def truncated_normal_(tensor, mean=0, std=1):
    """
    Fill the input tensor with values drawn from a truncated normal distribution.
    The values are truncated to [-2, 2] standard deviations.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to fill.
    mean : float, default=0
        Mean of the normal distribution.
    std : float, default=1
        Standard deviation of the normal distribution.
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def scale_limit(param, limit, bits_W):
    """
    Compute a scaling factor for WAGE weight initialization based on the
    quantization bit-width and the maximum absolute weight.

    Parameters
    ----------
    param : torch.nn.Parameter or torch.Tensor
        Parameter tensor with `.weight_scale` attribute to be set.
    limit : float
        Maximum absolute value for weight initialization.
    bits_W : int
        Bit width of the weight representation.

    Returns
    -------
    limit : float
        Scaled maximum absolute value for weight initialization.
    """
    beta = 1.5
    Wm = beta / (2 ** (bits_W - 1))
    scale = 2 ** round(np.log2(Wm / limit))
    scale = max(scale, 1.0)
    limit = max(Wm, limit)
    param.weight_scale = scale
    return limit


def wage_init_(tensor, bits_W, factor=2.0, mode="fan_in"):
    """
    Initialize a tensor for WAGE quantization using uniform distribution
    with limits determined by fan-in and quantization bit-width.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to initialize.
    bits_W : int
        Bit-width of the weight representation.
    factor : float, default=2.0
        Scaling factor for weight initialization.
    mode : str, default="fan_in"
        Initialization mode. Currently, only "fan_in" is supported.

    Raises
    ------
    NotImplementedError
        If mode is not "fan_in".
    ValueError
        If tensor has fewer than 2 dimensions.
    """
    if mode != "fan_in":
        raise NotImplementedError("support only wage normal")

    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError("tensor at least is 2d")
    elif dimensions == 2:
        fan_in = tensor.size(1)
    elif dimensions > 2:
        num_input_fmaps = tensor.size(1)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
    float_limit = math.sqrt(3 * factor / fan_in)
    quant_limit = scale_limit(tensor, float_limit, bits_W)#, name, scale_dict)
    tensor.data.uniform_(-quant_limit, quant_limit)