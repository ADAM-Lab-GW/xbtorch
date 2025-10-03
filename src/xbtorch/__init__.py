"""
XBTorch root package
=====================

This module provides the root API for XBTorch, including:

- The `XBParams` singleton class for global configuration.
- Helper functions to get/set parameters and initialize the library.
- Lists of supported activation layers, parameterized layers, and parameter-less layers.
"""

from functools import partial

import torch.nn as nn

import xbtorch
import xbtorch.quant.wage_qtorch as wage_qtorch

class XBParams:
    """
    Singleton class to store global XBTorch parameters.

    This class ensures a single global configuration dictionary that controls
    decomposition, device, quantization, weight ranges, and accelerators.

    Attributes
    ----------
    _global_dict : dict
        Dictionary storing all global parameters and flags.
    _wage_defaults : dict
        Default settings for WAGE quantization.
    """
    _instance = None
    _global_dict = {'initialized': False,
                    'decomposition_algorithm': None,
                    'device_type': None,
                    'pytorch_device': 'cpu',
                    'inference_accelerator': None,
                    'weight_range': (-1, 1),
                    }
    _wage_defaults = {
                    'wl_weight': 2, # 2 = ternary weights
                    'wl_activation': 8,
                    'wl_grad': 8,
                    'wl_error': 8,
                    'rounding_weight' : 'nearest',
                    'rounding_activation' : 'nearest',
                    'rounding_grad' : 'nearest',
                    'rounding_error' : 'nearest',
                   }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(XBParams, cls).__new__(cls)
        return cls._instance

    def set_var(self, key, value):
        self._global_dict[key] = value

    def get_var(self, key, default=None):
        return self._global_dict.get(key, default)
    
    def initialize(self, decomposition_algorithm=None, device_type=None, weight_range=(-1, 1), pytorch_device='cpu', wage_quantize=False, wage_params={}, inference_accelerator=None):
        """
        Initialize the XBTorch environment.

        Sets up decomposition algorithm, device, weight ranges, WAGE quantization,
        and optional inference accelerators. Also migrates tensors to the selected device.

        Parameters
        ----------
        decomposition_algorithm : xbtorch.decomposition.base.GenericDecomposition, optional
            Decomposition algorithm to use for layers (default is None).
        device_type : xbtorch.devices.base.GenericDevice, optional
            Hardware device abstraction (default is None).
        weight_range : tuple of float, optional
            Min and max allowed weights, default is (-1, 1).
        pytorch_device : str or torch.device, optional
            PyTorch device for tensor allocation (default 'cpu').
        wage_quantize : bool, optional
            Whether to enable WAGE quantization (default False).
        wage_params : dict, optional
            Overrides for WAGE quantization defaults.
        inference_accelerator : xbtorch.deployment.base.GenericAccelerator, optional
            Inference accelerator to use.

        Raises
        ------
        TypeError
            If provided decomposition_algorithm, device_type, or inference_accelerator
            is not of the expected type, or weight_range is invalid.
        """
        if decomposition_algorithm and not issubclass(type(decomposition_algorithm), xbtorch.decomposition.base.GenericDecomposition):
            raise TypeError("Invalid decomposition algorithm provided")
        if device_type and not issubclass(type(device_type), xbtorch.devices.base.GenericDevice):
            raise TypeError("Invalid device type provided")
        if wage_quantize:
            self._global_dict['wage_quantize'] = True
            self._global_dict['wage_params'] = {}
            for key, val in self._wage_defaults.items():
                if key not in wage_params: self._global_dict['wage_params'][key] = val # use defaults
                else: self._global_dict['wage_params'][key] = wage_params[key]
            # weight quantizer
            if self._global_dict['wage_params']['wl_weight'] == -1:
                self._global_dict['wage_params']['quantizer_weight'] = None
            else:
                self._global_dict['wage_params']['quantizer_weight'] = lambda x, scale: wage_qtorch.QW(
                    x, 
                    self._global_dict['wage_params']['wl_weight'], 
                    scale, 
                    mode=self._global_dict['wage_params']['rounding_weight']
                )

            # gradient quantizer
            if self._global_dict['wage_params']['wl_grad'] == -1:
                self._global_dict['wage_params']['quantizer_grad'] = None
                self._global_dict['wage_params']['grad_clip'] = None
            else:
                self._global_dict['wage_params']['quantizer_grad'] = lambda x, lr: wage_qtorch.QG(
                    x, 
                    self._global_dict['wage_params']['wl_grad'],
                    lr, 
                    mode=self._global_dict['wage_params']['rounding_grad']
                )
                self._global_dict['wage_params']['grad_clip'] = lambda x: wage_qtorch.C(x, self._global_dict['wage_params']['wl_weight'])

            # activation and error quantizer
            self._global_dict['wage_params']['quantizer_act_error'] = partial(
                wage_qtorch.WAGEQuantizer, 
                A_mode=self._global_dict['wage_params']['rounding_activation'], 
                E_mode=self._global_dict['wage_params']['rounding_error']
            )

        if weight_range and not (type(weight_range) == tuple or len(weight_range) != 2 or weight_range[0] >= weight_range[1]):
            raise TypeError("Invalid weight range provided")

        if inference_accelerator and not issubclass(type(inference_accelerator), xbtorch.deployment.base.GenericAccelerator):
            raise TypeError("Invalid accelerator algorithm provided")

        # Setting the variables
        self._global_dict['initialized'] = True
        self._global_dict['decomposition_algorithm'] = decomposition_algorithm
        self._global_dict['device_type'] = device_type
        self._global_dict['pytorch_device'] = pytorch_device
        self._global_dict['weight_range'] = weight_range
        self._global_dict['inference_accelerator'] = inference_accelerator

        # if a device_type was provided, migrate local tensors if needed
        if (pytorch_device != 'cpu'):
            if (device_type and issubclass(type(device_type), xbtorch.devices.base.TabularDevice)):
                device_type.set_G = device_type.set_G.to(pytorch_device)
                device_type.reset_G = device_type.reset_G.to(pytorch_device)

                device_type.set_dG = device_type.set_dG.to(pytorch_device)
                device_type.reset_dG = device_type.reset_dG.to(pytorch_device)

                device_type.set_cdf = device_type.set_cdf.to(pytorch_device)
                device_type.reset_cdf = device_type.reset_cdf.to(pytorch_device)

                device_type.min_conductance = device_type.min_conductance.to(pytorch_device)
                device_type.max_conductance = device_type.max_conductance.to(pytorch_device)

        # print('Initialization complete..\n')

def get_xbtorch_param(key, default=None):
    """
    Retrieve a global XBTorch parameter.

    Convenience function to access the `XBParams` singleton.

    Parameters
    ----------
    key : str
        Name of the parameter to retrieve.
    default : any, optional
        Value to return if key is not found (default None).

    Returns
    -------
    any
        The value of the requested parameter.
    """
    return XBParams().get_var(key, default)

def initialize(*args, **kwargs):
    """
    Initialize the XBTorch library.

    Convenience function that calls `XBParams.initialize`.

    See Also
    --------
    XBParams.initialize
    """
    XBParams().initialize(*args, **kwargs)

activation_types = (
    nn.ReLU,
    nn.Sigmoid,
    nn.Tanh,
    nn.LeakyReLU,
    nn.Softmax,
    nn.Softplus,
    nn.Softsign,
    nn.ELU,
    nn.PReLU,
    nn.SELU,
    nn.GELU,
    nn.Hardtanh,
    nn.Hardsigmoid,
    nn.Hardshrink,
    nn.Hardswish,
    nn.LogSigmoid,
    nn.SiLU,
    nn.Mish,
    nn.LogSoftmax,
)

layer_types = (
    nn.Linear,
    nn.Conv2d,
    nn.RNN,
    nn.LSTM,
    # future work
    # nn.Conv3d,
    # nn.ConvTranspose2d,
    # nn.ConvTranspose3d,
    # nn.BatchNorm1d,
    # nn.BatchNorm2d,
    # nn.BatchNorm3d,
    # nn.InstanceNorm1d,
    # nn.InstanceNorm2d,
    # nn.InstanceNorm3d,
    # nn.LayerNorm,
    # nn.GroupNorm,
    # nn.Embedding,
    # nn.GRU,
)

# Parameter-less layers
misc_types = (
    # Dropout
    nn.Dropout,
    nn.Dropout2d,
    nn.Dropout3d,
    # Pooling
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveMaxPool1d,
    nn.AdaptiveMaxPool2d,
    nn.AdaptiveMaxPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    # Flatten
    nn.Flatten,
)

__all__ = ['initialize', 'XBParams']