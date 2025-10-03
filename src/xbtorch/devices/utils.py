"""
Utility functions for device modeling, dataset generation, and interpolation.

This module provides low-level helpers for XBTorch device simulations, including:

- Conductance table search and nearest-neighbor functions.
- Stochastic rounding for pulse calculations.
- Device conductance sweeps for SET and RESET phases.
- Synthetic dataset generation for device characterization.
- Kriging-based interpolation and CDF construction for jump table models.

Functions
---------
- :func:`find_nearest` : Find nearest indices in a 1D array.
- :func:`find_nearest_2d` : Find nearest indices in a 2D array.
- :func:`stochastic_round` : Stochastically round values to nearest integer.
- :func:`sweep_conductance_full` : Sweep device conductances through SET and RESET trajectories.
- :func:`synthesize_G_dG_dataset` : Generate synthetic (G, dG) datasets for a device model.
- :func:`get_kriging_profiles` : Generate mean and standard deviation profiles using Kriging interpolation.
- :func:`construct_cdf` : Construct a CDF array mapping G to dG for jump table devices.
"""

import numpy as np
import random
import torch

from statistics import NormalDist
from smt.surrogate_models import KRG

def find_nearest(array, values):
    """
    Find the indices of the nearest elements in a 1D array for each value.

    Parameters
    ----------
    array : torch.Tensor
        1D tensor of reference values.
    values : torch.Tensor
        Tensor of values to match.

    Returns
    -------
    torch.Tensor
        Indices in `array` closest to each element in `values`.
    """
    # Expand dimensions for broadcasting
    array = array.unsqueeze(0)  
    values = values.unsqueeze(1) 
    abs_diff = torch.abs(array - values)
    indices = abs_diff.argmin(dim=1)
    return indices

def find_nearest_2d(array, values):
    """
    Find nearest indices in a 2D array for each value (broadcastable).

    Parameters
    ----------
    array : torch.Tensor
        1D or 2D tensor of reference values.
    values : torch.Tensor or array-like
        Values to match against the array.

    Returns
    -------
    torch.Tensor
        Indices of the nearest elements along the last axis.
    """
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
    """
    Stochastically round values to the nearest integer with probability proportional to fractional part.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of floating-point numbers.

    Returns
    -------
    torch.Tensor
        Stochastically rounded integers.
    """
    a = torch.floor(x)
    return a + ((x - a) > torch.rand_like(x)).float()

def sweep_conductance_full(device, group_param_idx=(0, 0)):
    """
    Sweep conductance trajectories of a device for SET and RESET operations.

    SET starts from minimum conductance and applies positive pulses.
    RESET starts from maximum conductance and applies negative pulses.

    Parameters
    ----------
    device : xbtorch.devices.GenericDevice
        An XBTorch device model.
    group_param_idx : tuple(int, int)
        Optional internal index used during neural network simulations.

    Returns
    -------
    tuple[list[float], list[float]]
        Conductance trajectories for SET and RESET as a tuple.
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
    """
    Generate a synthetic (G, dG) dataset for a device model.

    Iteratively samples delta conductance points from the device by applying pulses.

    Parameters
    ----------
    device : xbtorch.devices.GenericDevice
        The target device model.
    set : bool
        Sample in SET (True) or RESET (False) direction.
    num_points : int
        Number of points to generate.
    min_delta_ratio : float
        Minimum ratio deltaG/G to consider a point valid.
    disable_filtering : bool
        If True, include all points without filtering for physical realism.

    Returns
    -------
    np.ndarray
        Array of shape (num_points, 2) containing (G, dG) points.
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
    """
    Generate mean and standard deviation profiles using Kriging interpolation.

    Parameters
    ----------
    g : list[float]
        Initial conductances.
    dg : list[float]
        Conductance changes corresponding to `g`.
    axis_size : int
        Size of G and dG axes for the jump table.
    epsilon : float
        Small value to ensure non-zero standard deviation.
    kwargs : dict
        Additional parameters for SMT KRG surrogate model.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        G_axis, dG_axis, mean_profile, std_profile arrays.
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
    """
    Construct a cumulative distribution function (CDF) array for a jump table.

    Parameters
    ----------
    G_axis : np.ndarray
        The G axis of the jump table.
    dG_axis : np.ndarray
        The dG axis of the jump table.
    mean_profile : np.ndarray
        Mean conductance change profile.
    std_profile : np.ndarray
        Standard deviation profile.

    Returns
    -------
    np.ndarray
        2D array mapping G to dG probabilities (CDF).
    """
    cdf_array = np.zeros((len(G_axis), len(dG_axis)))
    for G_idx in range(len(G_axis)):
        dist = NormalDist(mu=mean_profile[G_idx], sigma=std_profile[G_idx])
        for dG_idx in range(len(dG_axis)):
            cdf_array[G_idx, dG_idx] = (dist.cdf(dG_axis[dG_idx]))

    return cdf_array