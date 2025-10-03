"""
Predefined device model presets for analog crossbar simulations.

This module provides ready-to-use instances of analytical and tabular
device models with fixed parameters for benchmarking, testing, or
hardware-aware training. Each preset wraps a specific device configuration
and provides attribute forwarding to the underlying device instance.

Classes
-------
- :class:`AnalyticalIdeal` : Ideal analytical device with zero variations.
- :class:`AnalyticalReal` : Realistic analytical device with typical c2c variations.
- :class:`TabularAnalyticalReal` : Synthetic tabular device mimicking realistic analytical behavior.
- :class:`TabularCompactFeFETKriging` : Compact FeFET tabular device with Kriging interpolation.
- :class:`TabularExperimentalFemFETKriging` : Experimental FeFET tabular device based on measured data.

Notes
-----
All preset devices wrap the corresponding :class:`AnalyticalDevice` or
:class:`TabularDevice` and forward attribute access using `__getattr__`
and `__setattr__`. This allows presets to behave like normal device
instances while keeping the preset parameters fixed.
"""

from .base import AnalyticalDevice, TabularDevice
from importlib.resources import files

class AnalyticalIdeal(AnalyticalDevice):
    """
    Ideal analytical device with zero device-to-device (d2d) and
    cycle-to-cycle (c2c) variations.

    Conductance range: 3 nS to 38 nS.
    Nonlinearity: set to zero.
    Maximum pulse level: 500.
    """

    def __init__(self):
        self._device = AnalyticalDevice(min_conductance=3e-9, max_conductance=38e-9, d2d_var=0.0, c2c_var=0.0, nonlinearity_set=0, nonlinearity_reset=0, max_level=500)

    def __getattr__(self, name):
        return getattr(self._device, name)
    
    def __setattr__(self, name, value):
        if name != "_device": setattr(self._device, name, value)
        else: super().__setattr__(name, value)

class AnalyticalReal(AnalyticalDevice):
    """
    Realistic analytical device with typical cycle-to-cycle variations.

    Conductance range: 3 nS to 38 nS.
    Nonlinearity: set = 2.4, reset = -4.88.
    c2c variation: 0.035.
    Maximum pulse level: 500.
    """

    def __init__(self):
        self._device = AnalyticalDevice(min_conductance=3e-9, max_conductance=38e-9, d2d_var=0.0, c2c_var=0.035, nonlinearity_set=2.4, nonlinearity_reset=-4.88, max_level=500)

    def __getattr__(self, name):
        return getattr(self._device, name)

    def __setattr__(self, name, value):
        if name != "_device": setattr(self._device, name, value)
        else: super().__setattr__(name, value)

class TabularAnalyticalReal(TabularDevice):
    """
    Synthetic tabular device simulating realistic analytical behavior.

    Uses predefined target tables with linear mean and constant standard
    deviation. Equivalent to NeuroSim synthetic dataset with 10,000 samples.
    Maximum pulse level: 500.
    """

    def __init__(self):
        model_name = 'synthetic/target_lc_ns_sd1.2_10000' # target tables, linear mean, constant standard deviation (1.2 nS) profile, equivalent to neurosim, constructed with 10,000 data points
        self._device = TabularDevice(reset_g=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_reset_axis_level_G.txt'), 
                                     reset_dg=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_reset_axis_level_dG.txt'), 
                                     reset_cdf=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_reset_cdf.txt'), 
                                     set_g=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_set_axis_level_G.txt'), 
                                     set_dg=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_set_axis_level_dG.txt'), 
                                     set_cdf=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_set_cdf.txt'), 
                                     max_level=500)

    def __getattr__(self, name):
        return getattr(self._device, name)

    def __setattr__(self, name, value):
        if name != "_device": setattr(self._device, name, value)
        else: super().__setattr__(name, value)
    

# Tabular FeFET device model presets from https://ieeexplore.ieee.org/abstract/document/10932692
class TabularCompactFeFETKriging(TabularDevice):
    """
    Compact FeFET tabular device using Kriging interpolation.

    Parameters
    ----------
    name : str
        Name encoding voltage and standard deviation, e.g., 'fefet_reset_1.2V_32_SD'.

    Notes
    -----
    Uses 10,000 data points to construct the tabular model, consistent with
    synthetic Neuromorphic simulations.
    Maximum pulse level: 32.
    """

    def __init__(self, name):
        filename = name.split('_')
        vgs, sd = filename[-3], filename[-1]
        model_name = 'fefet/Kriging' # target tables, linear mean, constant standard deviation (1.2 nS) profile, equivalent to neurosim, constructed with 10,000 data points
        self._device = TabularDevice(reset_g=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_reset{vgs}V_{sd}_SD_axis_level_G.txt'), 
                                     reset_dg=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_reset_axis_level_dG.txt'), 
                                     reset_cdf=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_reset{vgs}V_{sd}_SD_cdf.txt'), 
                                     set_g=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_set{vgs}V_{sd}_SD_axis_level_G.txt'), 
                                     set_dg=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_set_axis_level_dG.txt'), 
                                     set_cdf=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_set{vgs}V_{sd}_SD_cdf.txt'), 
                                     max_level=32)

    def __getattr__(self, name):
        return getattr(self._device, name)

    def __setattr__(self, name, value):
        if name != "_device": setattr(self._device, name, value)
        else: super().__setattr__(name, value) 

class TabularExperimentalFemFETKriging(TabularDevice):
    """
    Experimental FeFET tabular device model based on measured data.

    Parameters
    ----------
    name : str
        Name encoding gate voltage, e.g., 'femfet_vg3'.

    Notes
    -----
    Uses experimental data for reset/set conductances, dG, and CDF tables.
    Maximum pulse level: 50.
    """
    
    def __init__(self, name):
        filename = name.split('_')
        vg = filename[-1]
        model_name = 'experimental/TNANO_out3/tab_femfet_experimental_out3' # target tables, linear mean, constant standard deviation (1.2 nS) profile, equivalent to neurosim, constructed with 10,000 data points
        self._device = TabularDevice(reset_g=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_reset_vg{vg}_axis_level_G.txt'), 
                                     reset_dg=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_reset_vg{vg}_axis_level_dG.txt'), 
                                     reset_cdf=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_reset_vg{vg}_cdf.txt'), 
                                     set_g=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_set_vg{vg}_axis_level_G.txt'), 
                                     set_dg=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_set_vg{vg}_axis_level_dG.txt'), 
                                     set_cdf=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_set_vg{vg}_cdf.txt'), 
                                     max_level=50)

    def __getattr__(self, name):
        return getattr(self._device, name)

    def __setattr__(self, name, value):
        if name != "_device": setattr(self._device, name, value)
        else: super().__setattr__(name, value)

if __name__ == '__main__':
    pass