import numpy as np
import random
import torch

from statistics import NormalDist
from smt.surrogate_models import KRG

def find_nearest(array, values):
    # Expand dimensions for broadcasting
    array = array.unsqueeze(0)  
    values = values.unsqueeze(1) 
    abs_diff = torch.abs(array - values)
    indices = abs_diff.argmin(dim=1)
    return indices

def find_nearest_2d(array, values):
    # Expand dimensions for broadcasting
    if (array.dim() == 1): array = array.unsqueeze(0)

    array = array.unsqueeze(1)  

    if isinstance(values, torch.Tensor):
        values = values.to(device=array.device, dtype=array.dtype)
    else:
        values = torch.as_tensor(values, device=array.device, dtype=array.dtype)

    values = values.unsqueeze(1).unsqueeze(2)

    abs_diff = torch.abs(array - values)  
    indices = abs_diff.argmin(dim=2).squeeze(1)

    return indices

def stochastic_round(x):
    a = torch.floor(x)
    return a + ((x - a) > torch.rand_like(x)).float()

def sweep_conductance_full(device, group_param_idx=(0, 0)):
    """Sweep conductance trajectories of a given device model for SET and RESET phases
    For SET, start at the device's minimum conductance, applying positive direction pulses and reading back device conductances.
    For RESET, start at the device's maximum conductance, applying opposite direction pulses and reading back device conductances 

    Parameters
    ----------
    device : xbtorch.devices.GenericDevice
        An XBTorch device model object
    group_param_idx : tuple(int, int)
        Optional parameter used internally by XBTorch during neural network simulations. This should not be used when explicitly sweeping a device model.
    Returns
    -------
    (set_Gs, reset_Gs) : tuple[list[float], list[float]]
        Conductance trajectories for SET and RESET phases as a tuple.
    """
    set_Gs = [device.min_conductance_set]
    G = set_Gs[0]
    for i in range(device.max_level):
        G = device.write(torch.Tensor([G]), numPulse=torch.Tensor([1]), group_param_idx=group_param_idx).item()
        set_Gs.append(G)

    reset_Gs = [device.max_conductance_reset]
    G = reset_Gs[0]
    for i in range(device.max_level):
        G = device.write(torch.Tensor([G]), numPulse=torch.Tensor([-1]), group_param_idx=group_param_idx).item()
        reset_Gs.append(G)

    return set_Gs, reset_Gs

def synthesize_G_dG_dataset(device, set=True, num_points=1000, min_delta_ratio=1/1000, disable_filtering=False):

    """Synthetic Device Dataset Generation Algorithm. Iteratively samples delta conductance points uniformly within valid initial conductance values in the device, optionally checking for physical realism.
    Analogous to Algorithm 2 from: Device Modeling Bias in ReRAM-based Neural Network Simulations, https://ieeexplore.ieee.org/abstract/document/10024104
    Instead of using known mean and standard deviation profiles, we just use the device write operation, making the sampling device model-agnostic.

    Parameters
    ----------
    device : xbtorch.devices.GenericDevice
        An XBTorch device model object
    set : boolean
        Whether or not the sampling should be done in the positive direction (set) or negative direction (reset)
    num_points : int
        Total number of voltage pulses to apply to the device
    min_delta_ratio : float
        Minimum ratio (deltaG / G) must exceed in order for a sampled point to be considered valid
    disable_filtering : bool
        Whether or not to filter sampled candidate points for physical realism. If False, all sampled points are included in the final list.
    Returns
    -------
    dataset[num_points] : list[list[2]]
        Dataset D containing num_points many (initial conductance, delta conductance / pulse) or (G, dG) data points.
    """

    dataset = np.zeros((num_points, 2)) # generate (G, dG) dataset with total points specified by `num_points`
    numPulse = 1 if set else -1
    for point in range(num_points):
        valid = False
        while (not valid):
            G = random.uniform(device.min_conductance, device.max_conductance) # generate a random conductance value from possible device range
            newG = device.write(torch.Tensor([G]), numPulse=torch.Tensor([numPulse])).item()
            deltaG = newG - G
            # additional constraints can be added here for experimental realism
            if (disable_filtering): valid = True
            elif (newG >= device.min_conductance and newG <= device.max_conductance and abs(deltaG/G) > min_delta_ratio):  valid = True

            dataset[point] = [G, deltaG]

    return dataset

def get_kriging_profiles(g, dg, axis_size=100, epsilon=1e-10, **kwargs):

    """Generate mean and standard deviation jump table model profiles based on provided yt(xt) data using Kriging interpolation.
    Inspiration: 

    Parameters
    ----------
    g : list[float]
        List of initial device conductances
    dg : list[float]
        List of device conductance changes / applied voltage pulse (corresponding to initial conductances in g)
    axis_size : int
        Total size of the G and dG axes that the final jump table must have (dictates the size of the CDF array)
    min_delta_ratio : float
        Minimum ratio (deltaG / G) must exceed in order for a sampled point to be considered valid
    kwargs : dict
        Arguments for SMT's kriging model. See KRG's documentation for full details: https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models/gpr/krg.html
    Returns
    -------
    G_axis[axis_size] : list[float]
        The G axis of the jump table model
    dG_axis[axis_size] : list[float]
        The dG axis of the jump table model
    mean_profile[axis_size] : list[float]
        The overall mean profile of the jump table (as a function of g)
    std_profile[axis_size] : list[float]
        The overall std profile of the jump table (as a function of g)
    """

    # sanity checks
    assert (len(g) == len(dg))

    defaults = {
        'theta0': [1e-2],
        'noise0': [1]
    }

    combined_kwargs = {**defaults, **kwargs}

    sm = KRG(**combined_kwargs) # may pass noise0 and other params
    sm.set_training_values(g, dg)
    sm.train()

    dg_max = np.max(abs(dg))
    G_axis = np.linspace(min(g), max(g), axis_size)
    dG_axis = np.linspace(-dg_max, +dg_max, axis_size) # can be kept same/different for different models, would need absolute values 
    
    # mean profile
    mean_profile = sm.predict_values(G_axis)
    mean_profile = np.ravel(mean_profile)

    # std profile
    s2 = sm.predict_variances(G_axis)
    std_profile = np.sqrt(s2)
    std_profile = np.ravel(std_profile) + epsilon

    return G_axis, dG_axis, mean_profile, std_profile

def construct_cdf(G_axis, dG_axis, mean_profile, std_profile):

    """Generate a CDF array for a jump table model given its G, dG axes and mean, std profiles.
    Inspiration: 

    Parameters
    ----------
    G_axis[axis_size] : list[float]
        The G axis of the jump table model
    dG_axis[axis_size] : list[float]
        The dG axis of the jump table model
    cdf_array[axis_size, axis_size] : list[list[float]]
        The CDF array mapping the G axis to the dG axis
    mean_profile[axis_size] : list[float]
        The overall mean profile of the jump table (as a function of g)
    std_profile[axis_size] : list[float]
        The overall std profile of the jump table (as a function of g)
    Returns
    -------
    cdf_array[axis_size, axis_size] : list[list[float]]
        The CDF array mapping the G axis to the dG axis
    """

    cdf_array = np.zeros((len(G_axis), len(dG_axis)))

    for G_idx in range(len(G_axis)):
        dist = NormalDist(mu=mean_profile[G_idx], sigma=std_profile[G_idx])
        for dG_idx in range(len(dG_axis)):
            cdf_array[G_idx, dG_idx] = (dist.cdf(dG_axis[dG_idx]))

    return cdf_array