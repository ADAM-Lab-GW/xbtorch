"""
Base device classes for crossbar simulations.

This module provides abstract and concrete device models used to
simulate non-ideal analog memory devices. It defines the
`GenericDevice` base class and two primary implementations:

- :class:`AnalyticalDevice` : Nonlinear device model using analytical formulas.
- :class:`TabularDevice` : Empirical device model using tabulated measurements.

These classes handle conductance-to-weight conversion, pulse-based writes,
device-to-device (d2d) and cycle-to-cycle (c2c) variations, and nonlinear
weight update behavior.

Notes
-----
All device classes assume that `weight_range` is defined globally via
:func:`get_xbtorch_param`. The write methods modify tensors in-place
and enforce device conductance bounds.
"""

import numpy as np
import  torch
import abc
from importlib.resources import files # compatible with Python 3.10 - https://setuptools.pypa.io/en/latest/userguide/datafiles.html
import random

from .. import get_xbtorch_param
from .utils import stochastic_round, find_nearest, find_nearest_2d

class GenericDevice(metaclass=abc.ABCMeta):
    """
    Abstract base class for analog memory devices.

    Handles conversion between software weights and device conductance,
    gradient-to-pulse conversion, and caching of device parameters.

    Parameters
    ----------
    min_conductance : float, optional
        Minimum conductance of the device (used for both set and reset).
    max_conductance : float, optional
        Maximum conductance of the device (used for both set and reset).
    min_conductance_set : float, optional
        Minimum conductance during set operation.
    min_conductance_reset : float, optional
        Minimum conductance during reset operation.
    max_conductance_set : float, optional
        Maximum conductance during set operation.
    max_conductance_reset : float, optional
        Maximum conductance during reset operation.

    Raises
    ------
    ValueError
        If insufficient conductance bounds are provided.
    """

    def __init__(self, min_conductance=None, max_conductance=None, min_conductance_set=None, min_conductance_reset=None, max_conductance_set=None, max_conductance_reset=None):
        if (min_conductance and max_conductance):
            self.min_conductance_set, self.min_conductance_reset = min_conductance, min_conductance
            self.max_conductance_set, self.max_conductance_reset,  = max_conductance, max_conductance
        elif (min_conductance_set and min_conductance_reset and max_conductance_set and max_conductance_reset):
            self.min_conductance_set, self.min_conductance_reset = min_conductance_set, min_conductance_reset
            self.max_conductance_set, self.max_conductance_reset,  = max_conductance_set, max_conductance_reset
        else:
            raise ValueError("")

        # Make min/max values symmetrical
        self.min_conductance = max(self.min_conductance_set, self.min_conductance_reset)
        self.max_conductance = min(self.max_conductance_set, self.max_conductance_reset)
    
    def reset_cached_params(self):
        """
        Reset cached device parameters.
        """
        self.params = {}

    @abc.abstractmethod
    def write(self, G, pulse):
        """
        Abstract method to apply pulses to update conductance.

        Parameters
        ----------
        G : torch.Tensor
            Conductance tensor to update.
        pulse : torch.Tensor
            Number of pulses to apply.
        """
        pass

    def weight_to_conductance(self, weight):
        """
        Convert a software weight to device conductance.

        Parameters
        ----------
        weight : torch.Tensor
            Software weight tensor.

        Returns
        -------
        torch.Tensor
            Corresponding conductance tensor.
        """
        weight_range = get_xbtorch_param('weight_range')
        return (weight - weight_range[0]) / (weight_range[1] - weight_range[0]) * (self.max_conductance - self.min_conductance) + self.min_conductance

    def conductance_to_weight(self, conductance):
        """
        Convert device conductance back to software weight.

        Parameters
        ----------
        conductance : torch.Tensor
            Conductance tensor.

        Returns
        -------
        torch.Tensor
            Corresponding software weight tensor.
        """
        weight_range = get_xbtorch_param('weight_range')
        return (conductance - self.min_conductance) / (self.max_conductance - self.min_conductance) * (weight_range[1] - weight_range[0]) + weight_range[0]

    def gradient_to_pulse(self, gradient):
        """
        Convert a weight gradient to the number of device pulses using stochastic rounding.

        Parameters
        ----------
        gradient : torch.Tensor
            Weight gradient tensor.

        Returns
        -------
        torch.Tensor
            Integer tensor of pulses.
        """
        weight_range = get_xbtorch_param('weight_range')
        pulse = stochastic_round( (gradient / (weight_range[1] - weight_range[0])) * self.max_level)
        return pulse

class AnalyticalDevice(GenericDevice):
    """
    Analytical nonlinear device model.

    Uses exponential nonlinear weight update functions and supports
    device-to-device (d2d) and cycle-to-cycle (c2c) variations.

    Parameters
    ----------
    min_conductance : float
        Minimum conductance.
    max_conductance : float
        Maximum conductance.
    d2d_var : float
        Device-to-device variation.
    c2c_var : float
        Cycle-to-cycle variation.
    nonlinearity_set : float
        Nonlinearity of set operation.
    nonlinearity_reset : float
        Nonlinearity of reset operation.
    max_level : int
        Maximum number of pulse levels.
    """

    data_dir = 'analytical'

    def __init__(self, min_conductance, max_conductance, d2d_var, c2c_var, nonlinearity_set, nonlinearity_reset, max_level):
        super().__init__(min_conductance, max_conductance)

        # Device parameters
        self.d2d_var = d2d_var
        self.c2c_var = c2c_var
        self.nonlinearity_set = nonlinearity_set if nonlinearity_set != 0.0 else 1e-9
        self.nonlinearity_reset = nonlinearity_reset if nonlinearity_set != 0.0 else 1e-9
        self.max_level = max_level

        self.reset_cached_params()

    def reset_cached_params(self):
        """
        Reset cached parameters for pulse-conductance calculations.
        """
        self.params = {'param_A_set': {}, 'param_A_reset': {}, 'param_B_set': {}, 'param_B_reset': {}}

    def set_c2c_var(self, var):
        """Set cycle-to-cycle variation."""
        self.c2c_var = var

    def set_d2d_var(self, var):
        """Set device-to-device variation."""
        self.d2d_var = var

    def set_nonlinearity_set(self, nl_set):
        """Set nonlinearity for set operation."""
        self.nonlinearity_set = nl_set if nl_set != 0.0 else 1e-9

    def set_nonlinearity_reset(self, nl_reset):
        """Set nonlinearity for reset operation."""
        self.nonlinearity_reset = nl_reset if nl_reset != 0.0 else 1e-9

    # Get the conductance in the nonlinear weight function given a pulse position xPulse
    def pulse_to_conductance(self, xPulse, A, B):
        """
        Map pulse position to conductance using nonlinear function.

        Parameters
        ----------
        xPulse : torch.Tensor
            Pulse position tensor.
        A : torch.Tensor
            Parameter A.
        B : torch.Tensor
            Parameter B.

        Returns
        -------
        torch.Tensor
            Conductance tensor.
        """
        return B*(1-torch.exp(-xPulse/A)) + self.min_conductance

    # Inverse nonlinear weight function
    # get the pulse position based on the conductance of the weight update curve
    def conductance_to_pulse(self, G, A, B):
        """
        Inverse of pulse_to_conductance: map conductance to pulse position.

        Parameters
        ----------
        G : torch.Tensor
            Conductance tensor.
        A : torch.Tensor
            Parameter A.
        B : torch.Tensor
            Parameter B.

        Returns
        -------
        torch.Tensor
            Pulse position tensor.
        """
        return -A*torch.log(1 - (G-self.min_conductance)/B)

    def get_param_A(self, nonlinearity):
        """
        Get parameter A from nonlinearity using tabulated data (based on DNN + NeuroSim v2).

        Parameters
        ----------
        nonlinearity : torch.Tensor
            Nonlinearity tensor.

        Returns
        -------
        torch.Tensor
            Parameter A tensor.
        """
        index = (torch.abs(nonlinearity) * 100).long() - 1  # Use int64 for indices
        index = torch.where(index < 0, torch.zeros_like(index), index)
        index = torch.where(index > 899, torch.full_like(index, 899), index)
        sign = torch.sign(nonlinearity)
        data = torch.tensor(np.loadtxt(files('xbtorch.libdata').joinpath(f'{self.data_dir}/paramAdata.txt')), dtype=torch.float32)
        ADim = list(index.shape) + [data.shape[0]]
        lookupdata = data.repeat(*ADim[:-1], 1)
        y = torch.take_along_dim(lookupdata, index.unsqueeze(-1), dim=-1).squeeze(-1)
        A = sign * y
        return A

    def get_param_B(self, A):
        """
        Compute parameter B from A and max/min conductance.

        Parameters
        ----------
        A : torch.Tensor
            Parameter A tensor.

        Returns
        -------
        torch.Tensor
            Parameter B tensor.
        """
        return (self.max_conductance - self.min_conductance) / (1 - torch.exp(-self.max_level/A))
    
    def write(self, G, numPulse, group_param_idx=(0, 0)):
        """
        Apply pulses to update the device conductance tensor.

        Parameters
        ----------
        G : torch.Tensor
            Conductance tensor to update.
        numPulse : torch.Tensor
            Number of pulses for each element.
        group_param_idx : tuple, optional
            Index used for parameter caching (default: (0, 0)).

        Returns
        -------
        torch.Tensor
            Updated conductance tensor.
        """
        if (group_param_idx not in self.params['param_A_set']):
            # caching parameters for making write faster
            d2dVariation = torch.normal(torch.zeros_like(G), self.d2d_var*torch.ones_like(G))
            NL_set = torch.ones_like(G)*self.nonlinearity_set+d2dVariation
            NL_reset = torch.ones_like(G)*self.nonlinearity_reset+d2dVariation
            self.params['param_A_set'][group_param_idx] = self.get_param_A(NL_set)*self.max_level
            self.params['param_A_reset'][group_param_idx] = self.get_param_A(NL_reset)*self.max_level
            self.params['param_B_set'][group_param_idx] = self.get_param_B(self.params['param_A_set'][group_param_idx])
            self.params['param_B_reset'][group_param_idx] = self.get_param_B(self.params['param_A_reset'][group_param_idx])

        # constant look-ups
        self.param_A_set =  self.params['param_A_set'][group_param_idx]
        self.param_A_reset = self.params['param_A_reset'][group_param_idx]

        self.param_B_set = self.params['param_B_set'][group_param_idx]
        self.param_B_reset = self.params['param_B_reset'][group_param_idx]

        # Ensure numPulse is an integer tensor
        numPulse = numPulse.int()

        # Masks for positive and negative numPulse
        positive_mask = numPulse > 0
        negative_mask = numPulse < 0

        if positive_mask.any():
            pos_xPulse = self.conductance_to_pulse(G[positive_mask], self.param_A_set[positive_mask], self.param_B_set[positive_mask])
            for _ in range(numPulse[positive_mask].max()):
                step_mask = (numPulse > 0)
                if step_mask.any():
                    pos_xPulse[step_mask[positive_mask]] = pos_xPulse[step_mask[positive_mask]] + 1
                    G[positive_mask] = self.pulse_to_conductance(pos_xPulse, self.param_A_set[positive_mask], self.param_B_set[positive_mask])
                    c2c = torch.normal(torch.zeros_like(G[positive_mask]), self.c2c_var * (self.max_conductance - self.min_conductance) * torch.ones_like(G[positive_mask]))
                    G[positive_mask] = G[positive_mask] + c2c
                    G[positive_mask] = torch.where(G[positive_mask] > self.max_conductance, self.max_conductance, G[positive_mask])
                    G[positive_mask] = torch.where(G[positive_mask] < self.min_conductance, self.min_conductance, G[positive_mask])
                    numPulse[step_mask] -= 1

        if negative_mask.any():
            neg_xPulse = self.conductance_to_pulse(G[negative_mask], self.param_A_reset[negative_mask], self.param_B_reset[negative_mask])
            for _ in range(-numPulse[negative_mask].min()):
                step_mask = (numPulse < 0)
                if step_mask.any():
                    neg_xPulse[step_mask[negative_mask]] = neg_xPulse[step_mask[negative_mask]] - 1
                    G[negative_mask] = self.pulse_to_conductance(neg_xPulse, self.param_A_reset[negative_mask], self.param_B_reset[negative_mask])
                    c2c = torch.normal(torch.zeros_like(G[negative_mask]), self.c2c_var * (self.max_conductance - self.min_conductance) * torch.ones_like(G[negative_mask]))
                    G[negative_mask] = G[negative_mask] + c2c
                    G[negative_mask] = torch.where(G[negative_mask] > self.max_conductance, self.max_conductance, G[negative_mask])
                    G[negative_mask] = torch.where(G[negative_mask] < self.min_conductance, self.min_conductance, G[negative_mask])
                    numPulse[step_mask] += 1

        return G

class TabularDevice(GenericDevice):
    """
    Device model using tabulated empirical measurements.

    Uses gTable, dGTable, and cdfTable to simulate stochastic
    weight updates with cycle-to-cycle variations.

    Parameters
    ----------
    reset_g : str
        Path to reset conductance table.
    reset_dg : str
        Path to reset dG table.
    reset_cdf : str
        Path to reset CDF table.
    set_g : str
        Path to set conductance table.
    set_dg : str
        Path to set dG table.
    set_cdf : str
        Path to set CDF table.
    max_level : int
        Maximum pulse level.
    """

    data_dir = 'tabular'

    def __init__(self, reset_g, reset_dg, reset_cdf, set_g, set_dg, set_cdf, max_level):

        self.reset_G, self.set_G = np.loadtxt(reset_g, dtype=np.float32), np.loadtxt(set_g, dtype=np.float32) # Read gTable
        self.reset_dG, self.set_dG = np.loadtxt(reset_dg, dtype=np.float32), np.loadtxt(set_dg, dtype=np.float32) # Read dgTable
        self.reset_cdf, self.set_cdf = np.loadtxt(reset_cdf, dtype=np.float32), np.loadtxt(set_cdf, dtype=np.float32)  # Read cdfTable

        # convert to pytorch tensors
        self.set_G = torch.from_numpy(self.set_G)
        self.reset_G = torch.from_numpy(self.reset_G)

        self.set_dG = torch.from_numpy(self.set_dG)
        self.reset_dG = torch.from_numpy(self.reset_dG)

        self.set_cdf = torch.from_numpy(self.set_cdf)
        self.reset_cdf = torch.from_numpy(self.reset_cdf)

        min_conductance_reset = self.reset_G[0]
        max_conductance_reset = self.reset_G[-1]

        min_conductance_set = self.set_G[0]
        max_conductance_set = self.set_G[-1]

        super().__init__(min_conductance_set=min_conductance_set, max_conductance_set=max_conductance_set, min_conductance_reset=min_conductance_reset, max_conductance_reset=max_conductance_reset)
        self.max_level = max_level

    def write(self, G, numPulse, group_param_idx=None):
        """
        Apply pulses using tabulated device behavior.

        Parameters
        ----------
        G : torch.Tensor
            Conductance tensor to update.
        numPulse : torch.Tensor
            Number of pulses for each element.
        group_param_idx : None
            Not used for TabularDevice.

        Returns
        -------
        torch.Tensor
            Updated conductance tensor.
        """
        # Ensure numPulse is an integer tensor
        numPulse = numPulse.int()

        positive_mask = numPulse > 0
        negative_mask = numPulse < 0

        if positive_mask.any():
            # Process positive numPulse elements
            for _ in range(numPulse[positive_mask].max()):
                step_mask = (numPulse > 0)
                if step_mask.any():
                    near_G = find_nearest(self.set_G, G[step_mask])
                    prob = torch.rand(near_G.shape)
                    cdf = find_nearest_2d(self.set_cdf[near_G], prob)
                    dG = self.set_dG[cdf]
                    G[step_mask] = G[step_mask] + dG
                    G[step_mask] = torch.where(G[step_mask] > self.max_conductance, self.max_conductance, G[step_mask])
                    G[step_mask] = torch.where(G[step_mask] < self.min_conductance, self.min_conductance, G[step_mask])
                    numPulse[step_mask] -= 1

        if negative_mask.any():
            # Process negative numPulse elements
            for _ in range(-numPulse[negative_mask].min()):
                step_mask = (numPulse < 0)
                if step_mask.any():
                    near_G = find_nearest(self.reset_G, G[step_mask])
                    prob = torch.rand(near_G.shape)
                    cdf = find_nearest_2d(self.reset_cdf[near_G], prob)
                    dG = self.reset_dG[cdf]
                    G[step_mask] = G[step_mask] + dG
                    G[step_mask] = torch.where(G[step_mask] > self.max_conductance, self.max_conductance, G[step_mask])
                    G[step_mask] = torch.where(G[step_mask] < self.min_conductance, self.min_conductance, G[step_mask])
                    numPulse[step_mask] += 1

        return G

    
if __name__ == '__main__':
    pass