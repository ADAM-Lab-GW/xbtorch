import matplotlib.pyplot as plt
import numpy as np

from xbtorch.devices.presets import AnalyticalIdeal, AnalyticalReal, TabularAnalyticalReal
from xbtorch.devices.utils import synthesize_G_dG_dataset, sweep_conductance_full

def plot_G_vs_dG(device, num_points):
    '''
    
    Alternately, dG/dp can be computed using finite element differences on utils.sweep_conductance_full data (G(pulse), pulse)
    '''
    set_dataset = synthesize_G_dG_dataset(device, set=True, num_points=num_points, disable_filtering=False)
    reset_dataset = synthesize_G_dG_dataset(device, set=False, num_points=num_points, disable_filtering=False)

    plt.figure()
    plt.scatter(set_dataset[:, 0], set_dataset[:, 1], label='SET', s=10, alpha=0.5)
    plt.legend()
    plt.xlabel('Conductance (S)')
    plt.ylabel('Conductance / pulse (S)')

    plt.figure()
    plt.scatter(reset_dataset[:, 0], reset_dataset[:, 1], label='RESET', s=10, alpha=0.5)
    plt.legend()
    plt.xlabel('Conductance (S)')
    plt.ylabel('Conductance / pulse (S)')

    plt.ylim(0, device.min_conductance)

    return set_dataset, reset_dataset

# Analytical Device Models
deviceIdeal = AnalyticalIdeal()
deviceReal =  AnalyticalReal()

plot_G_vs_dG(deviceIdeal, num_points=1000)
plot_G_vs_dG(deviceReal, num_points=1000)
# plt.show()

# Tabular Device Models
# tabularDev = TabularAnalyticalReal()
# plot_G_vs_dG(tabularDev, num_points=10000)
plt.show()