import matplotlib.pyplot as plt
import numpy as np

from xbtorch.devices.presets import AnalyticalIdeal, AnalyticalReal, TabularAnalyticalReal
from xbtorch.devices.base import AnalyticalDevice
from xbtorch.devices.utils import sweep_conductance_full

def plot_G_vs_pulse(device, cycles=1):

    set_Gs, reset_Gs = np.zeros((cycles, device.max_level+1)), np.zeros((cycles, device.max_level+1))

    for cycle in range(cycles): set_Gs[cycle], reset_Gs[cycle] = sweep_conductance_full(device)

    plt.figure()
    plt.plot(range(0, device.max_level+1), np.average(set_Gs, axis=0), label='SET')
    plt.fill_between(range(0, device.max_level+1), np.average(set_Gs, axis=0) - np.std(set_Gs, axis=0), np.average(set_Gs, axis=0) + np.std(set_Gs, axis=0), alpha=0.2)

    plt.plot(range(device.max_level, device.max_level*2+1), np.average(reset_Gs, axis=0), label='RESET')
    plt.fill_between(range(device.max_level, device.max_level*2+1), np.average(reset_Gs, axis=0) - np.std(reset_Gs, axis=0), np.average(reset_Gs, axis=0) + np.std(reset_Gs, axis=0), alpha=0.2)

    plt.legend()
    plt.xlabel('Pulse (#)')
    plt.ylabel('Conductance (S)')

    return set_Gs, reset_Gs

# Analytical Device Models
deviceIdeal = AnalyticalIdeal()
deviceReal =  AnalyticalReal()

# customized parameters can also be used
# Here's an example of a device with non-linearity but no cycle-to-cycle variability 
deviceRealAltered =  AnalyticalReal()
deviceRealAltered.c2c_var = 0.0

# Alternately, fitted parameters can be directly utilized as well
deviceCustom = AnalyticalDevice(min_conductance=3e-9, max_conductance=3.8e-9, d2d_var=0.05, c2c_var=0.02, nonlinearity_set=5, nonlinearity_reset=-5, max_level=32)

plot_G_vs_pulse(deviceIdeal, cycles=10)
plot_G_vs_pulse(deviceReal, cycles=10)
plot_G_vs_pulse(deviceRealAltered, cycles=10)
plot_G_vs_pulse(deviceCustom, cycles=10)
# plt.show()

# Tabular Device Models
tabularDev = TabularAnalyticalReal()
plot_G_vs_pulse(tabularDev, cycles=10)
plt.show()