import numpy as np
import  torch
import abc
from importlib.resources import files # compatible with Python 3.10 - https://setuptools.pypa.io/en/latest/userguide/datafiles.html
import random

from .. import get_xbtorch_param
from .utils import stochastic_round, find_nearest, find_nearest_2d

class GenericDevice(metaclass=abc.ABCMeta):
    def __init__(self, min_conductance, max_conductance):
        self.min_conductance = min_conductance
        self.max_conductance = max_conductance
    
    def reset_cached_params(self):
        self.params = {}

    @abc.abstractmethod
    def write(self, G, pulse):
        pass

    def weight_to_conductance(self, weight):
        weight_range = get_xbtorch_param('weight_range')
        return (weight - weight_range[0]) / (weight_range[1] - weight_range[0]) * (self.max_conductance - self.min_conductance) + self.min_conductance

    def conductance_to_weight(self, conductance):
        weight_range = get_xbtorch_param('weight_range')
        return (conductance - self.min_conductance) / (self.max_conductance-self.min_conductance) * (weight_range[1] - weight_range[0]) + weight_range[0]

    def gradient_to_pulse(self, gradient):
        weight_range = get_xbtorch_param('weight_range')
        pulse = stochastic_round( (gradient / (weight_range[1] - weight_range[0])) * self.max_level)
        return pulse

class AnalyticalDevice(GenericDevice):
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
        self.params = {'param_A_set': {}, 'param_A_reset': {}, 'param_B_set': {}, 'param_B_reset': {}}

    def set_c2c_var(self, var):
        self.c2c_var = var

    def set_d2d_var(self, var):
        self.d2d_var = var

    def set_nonlinearity_set(self, nl_set):
        self.nonlinearity_set = nl_set if nl_set != 0.0 else 1e-9

    def set_nonlinearity_reset(self, nl_reset):
        self.nonlinearity_reset = nl_reset if nl_reset != 0.0 else 1e-9

    # Get the conductance in the nonlinear weight function given a pulse position xPulse
    def pulse_to_conductance(self, xPulse, A, B):
        return B*(1-torch.exp(-xPulse/A)) + self.min_conductance

    # Inverse nonlinear weight function
    # get the pulse position based on the conductance of the weight update curve
    def conductance_to_pulse(self, G, A, B):
        return -A*torch.log(1 - (G-self.min_conductance)/B)

    def get_param_A(self, nonlinearity):
        '''
        Adapted from DNN + NeuroSim v2
        '''
        nonlinearity = nonlinearity.numpy()
        index = (np.abs(nonlinearity)*100).astype(int)-1
        index = np.where(index<0, np.zeros_like(index), index)
        index = np.where(index>899, np.ones_like(index)*899, index)
        sign = np.sign(nonlinearity)

        data = np.loadtxt(files('xbtorch.data').joinpath(f'{self.data_dir}/paramAdata.txt'), dtype=np.float32)

        # extend A table to 2d or 4d
        ADim = np.append(np.delete(index.shape,-1),1)
        lookupdata = np.tile(data,ADim)
        # find a value according to index from the extend table
        y = np.take_along_axis(lookupdata, index, axis=-1)
        A = sign * y
        return torch.from_numpy(A)

    def get_param_B(self, A):
        return (self.max_conductance - self.min_conductance) / (1 - torch.exp(-self.max_level/A))
    
    def write(self, G, numPulse, group_param_idx=(0, 0)):
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
    data_dir = 'tabular'
    def __init__(self, reset_g, reset_dg, reset_cdf, set_g, set_dg, set_cdf, max_level, preconstructed=True):

        if (not preconstructed):
            raise NotImplementedError("Only preconstructed tabular models are possible.")
        else:
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

            min_conductance = self.reset_G[0]
            max_conductance = self.reset_G[-1]

            # sanity checks
            assert min_conductance ==  self.set_G[0]
            assert max_conductance ==  self.set_G[-1]


        super().__init__(min_conductance, max_conductance)
        self.max_level = max_level

    def write(self, G, numPulse, group_param_idx=None):
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