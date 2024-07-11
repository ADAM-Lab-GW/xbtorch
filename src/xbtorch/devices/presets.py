from .base import AnalyticalDevice, TabularDevice
from importlib.resources import files

class AnalyticalIdeal(AnalyticalDevice):
    def __init__(self):
        self._device = AnalyticalDevice(min_conductance=3e-9, max_conductance=38e-9, d2d_var=0.0, c2c_var=0.0, nonlinearity_set=0, nonlinearity_reset=0, max_level=500)

    def __getattr__(self, name):
        return getattr(self._device, name)
    
    def __setattr__(self, name, value):
        if name != "_device": setattr(self._device, name, value)
        else: super().__setattr__(name, value)

class AnalyticalReal(AnalyticalDevice):
    def __init__(self):
        self._device = AnalyticalDevice(min_conductance=3e-9, max_conductance=38e-9, d2d_var=0.0, c2c_var=0.035, nonlinearity_set=2.4, nonlinearity_reset=-4.88, max_level=500)

    def __getattr__(self, name):
        return getattr(self._device, name)

    def __setattr__(self, name, value):
        if name != "_device": setattr(self._device, name, value)
        else: super().__setattr__(name, value)

class TabularAnalyticalReal(TabularDevice):
    def __init__(self):
        model_name = 'synthetic/target_lc_ns_sd1.2_10000' # target tables, linear mean, constant standard deviation (1.2 nS) profile, equivalent to neurosim, constructed with 10,000 data points
        self._device = TabularDevice(reset_g=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_reset_axis_level_G.txt'), 
                                     reset_dg=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_reset_axis_level_dG.txt'), 
                                     reset_cdf=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_reset_cdf.txt'), 
                                     set_g=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_set_axis_level_G.txt'), 
                                     set_dg=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_set_axis_level_dG.txt'), 
                                     set_cdf=files('xbtorch.libdata').joinpath(f'{super().data_dir}/{model_name}_set_cdf.txt'), 
                                     max_level=500,
                                     preconstructed=True)

    def __getattr__(self, name):
        return getattr(self._device, name)

    def __setattr__(self, name, value):
        if name != "_device": setattr(self._device, name, value)
        else: super().__setattr__(name, value)
    

# FeFET device models used for T-NANO

class TabularCompactFeFETKriging(TabularDevice):
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
                                     max_level=32,
                                     preconstructed=True)

    def __getattr__(self, name):
        return getattr(self._device, name)

    def __setattr__(self, name, value):
        if name != "_device": setattr(self._device, name, value)
        else: super().__setattr__(name, value)

class TabularExperimentalFemFETKriging(TabularDevice):
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
                                     max_level=50,
                                     preconstructed=True)

    def __getattr__(self, name):
        return getattr(self._device, name)

    def __setattr__(self, name, value):
        if name != "_device": setattr(self._device, name, value)
        else: super().__setattr__(name, value)

if __name__ == '__main__':
    pass