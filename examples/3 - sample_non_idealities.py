import matplotlib.pyplot as plt
import numpy as np

from xbtorch.devices.presets import AnalyticalIdeal, AnalyticalReal, TabularAnalyticalReal
from xbtorch.devices.utils import synthesize_G_dG_dataset, sweep_conductance_full

# TODO
# showcase device non-idealities
# cycle to cycle variability
# device to device variability
# finite conductance states
# showcase finite conductance states.. somehow
def plot_non_idealities(num_devices, cycles, non_linearities):

    # Analytical Device Models
    device =  AnalyticalReal()

    # device to device variability (alter non linearity internally)
    device.set_c2c_var(0.0)
    device.set_d2d_var(1.0)
    plt.figure()
    set_Gs, reset_Gs = np.zeros((num_devices, device.max_level+1)), np.zeros((num_devices, device.max_level+1))
    for dev_num in range(num_devices): 
        set_Gs[dev_num], reset_Gs[dev_num] = sweep_conductance_full(device, dev_num)
        plt.plot(range(0, device.max_level+1), set_Gs[dev_num])
        plt.plot(range(device.max_level, device.max_level*2+1), reset_Gs[dev_num])
        device.reset_cached_params()

    plt.xlabel('Pulse (#)')
    plt.ylabel('Conductance (S)')

    # cycle to cycle variability
    device.set_c2c_var(0.035)
    device.set_d2d_var(0.0)
    set_Gs, reset_Gs = np.zeros((cycles, device.max_level+1)), np.zeros((cycles, device.max_level+1))

    for cycle in range(cycles): 
        set_Gs[cycle], reset_Gs[cycle] = sweep_conductance_full(device)
        device.reset_cached_params() # not needed, but good practice to stay consistent

    plt.figure()
    plt.plot(range(0, device.max_level+1), np.average(set_Gs, axis=0), label='SET')
    plt.fill_between(range(0, device.max_level+1), np.average(set_Gs, axis=0) - np.std(set_Gs, axis=0), np.average(set_Gs, axis=0) + np.std(set_Gs, axis=0), alpha=0.2)

    plt.plot(range(device.max_level, device.max_level*2+1), np.average(reset_Gs, axis=0), label='RESET')
    plt.fill_between(range(device.max_level, device.max_level*2+1), np.average(reset_Gs, axis=0) - np.std(reset_Gs, axis=0), np.average(reset_Gs, axis=0) + np.std(reset_Gs, axis=0), alpha=0.2)

    # non linearity
    device.set_c2c_var(0.0)
    device.set_d2d_var(0.0)
    plt.figure()
    set_Gs, reset_Gs = np.zeros((num_devices, device.max_level+1)), np.zeros((num_devices, device.max_level+1))
    for idx in range(len(non_linearities)):
        device.set_nonlinearity_set(non_linearities[idx])
        device.set_nonlinearity_reset(-non_linearities[idx])
        set_Gs[idx], reset_Gs[idx] = sweep_conductance_full(device)
        plt.plot(range(0, device.max_level+1), set_Gs[idx])
        plt.plot(range(device.max_level, device.max_level*2+1), reset_Gs[idx])
        device.reset_cached_params()

    plt.xlabel('Pulse (#)')
    plt.ylabel('Conductance (S)')

    return

plot_non_idealities(num_devices=6, cycles=10, non_linearities=range(0, 6))
plt.show()