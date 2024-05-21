import numpy as np
import  torch
import abc
from importlib.resources import files # compatible with Python 3.10 - https://setuptools.pypa.io/en/latest/userguide/datafiles.html
from xbtorch.devices.utils import find_nearest
import random

class GenericDevice(metaclass=abc.ABCMeta):
    def __init__(self, min_conductance, max_conductance):
        self.min_conductance = min_conductance
        self.max_conductance = max_conductance

    @abc.abstractmethod
    def write(self):
        pass

class AnalyticalDevice(GenericDevice):
    data_dir = 'analytical'
    def __init__(self, min_conductance, max_conductance, d2d_var, c2c_var, nonlinearity_set, nonlinearity_reset, max_level, shape=(1)):
        super().__init__(min_conductance, max_conductance)

        # Device parameters
        self.d2d_var = d2d_var
        self.c2c_var = c2c_var
        self.nonlinearity_set = nonlinearity_set if nonlinearity_set != 0.0 else 1e-9
        self.nonlinearity_reset = nonlinearity_reset if nonlinearity_set != 0.0 else 1e-9
        self.max_level = max_level

        dims = torch.ones(shape)
        NL_set = torch.ones_like(dims)*self.nonlinearity_set+self.d2d_var
        NL_reset = torch.ones_like(dims)*self.nonlinearity_reset+self.d2d_var

        self.param_A_set = torch.from_numpy(self.get_param_A(NL_set.cpu().numpy())*self.max_level) # TODO: remove conversions
        self.param_A_reset = torch.from_numpy(self.get_param_A(NL_reset.cpu().numpy())*self.max_level) # TODO: remove conversions

        self.param_B_set = self.get_param_B(self.param_A_set)
        self.param_B_reset = self.get_param_B(self.param_A_reset)

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
        index = (np.abs(nonlinearity)*100).astype(int)-1
        index = np.where(index<0, np.zeros_like(index), index)
        index = np.where(index>899, np.ones_like(index)*899, index)
        sign = np.sign(nonlinearity)

        data = np.loadtxt(files('xbtorch.data').joinpath(f'{self.data_dir}/paramAdata.txt'))

        # extend A table to 2d or 4d
        ADim = np.append(np.delete(index.shape,-1),1)
        lookupdata = np.tile(data,ADim)
        # find a value according to index from the extend table
        y = np.take_along_axis(lookupdata, index, axis=-1)
        A = sign * y
        return A

    def get_param_B(self, A):
        return (self.max_conductance - self.min_conductance) / (1 - torch.exp(-self.max_level/A))
    
    def write(self, G, numPulse):
        numPulse = int(numPulse)
        if (numPulse > 0):
            for _ in range(numPulse):
                xPulse = self.conductance_to_pulse(G, self.param_A_set, self.param_B_set)
                G = self.pulse_to_conductance(xPulse+1, self.param_A_set, self.param_B_set)
                c2c = torch.normal(torch.zeros_like(G), self.c2c_var*(self.max_conductance - self.min_conductance)*torch.ones_like(G))
                G = G + c2c
                if (G > self.max_conductance):
                    G = torch.Tensor([self.max_conductance])
                elif (G < self.min_conductance):
                    G = torch.Tensor([self.min_conductance])

        elif (numPulse < 0):
            for _ in range(-1*numPulse):
                xPulse = self.conductance_to_pulse(G, self.param_A_reset, self.param_B_reset)
                G = self.pulse_to_conductance(xPulse-1, self.param_A_reset, self.param_B_reset)
                c2c = torch.normal(torch.zeros_like(G), self.c2c_var*(self.max_conductance - self.min_conductance)*torch.ones_like(G))
                G = G + c2c
                if (G > self.max_conductance):
                    G = torch.Tensor([self.max_conductance])[0]
                elif (G < self.min_conductance):
                    G = torch.Tensor([self.min_conductance])[0]
        return G

class TabularDevice(GenericDevice):
    data_dir = 'tabular'
    def __init__(self, reset_g, reset_dg, reset_cdf, set_g, set_dg, set_cdf, max_level, preconstructed=True):

        if (not preconstructed):
            raise NotImplementedError("Only preconstructed tabular models are possible.")
        else:
            self.reset_G, self.set_G = np.loadtxt(reset_g), np.loadtxt(set_g) # Read gTable
            self.reset_dG, self.set_dG = np.loadtxt(reset_dg), np.loadtxt(set_dg) # Read dgTable
            self.reset_cdf, self.set_cdf = np.loadtxt(reset_cdf), np.loadtxt(set_cdf)  # Read cdfTable
        
            min_conductance = self.reset_G[0]
            max_conductance = self.reset_G[-1]

            # sanity checks
            assert min_conductance ==  self.set_G[0]
            assert max_conductance ==  self.set_G[-1]


        super().__init__(min_conductance, max_conductance)
        self.max_level = max_level

    def write(self, G, numPulse):
        '''
            Core method for tabular device models - given input conductance and no. of pulses, computes the output conductance with dG using jump tables

            X              : The original conductance (converted from network weight value)
            numPulse       : The number of pulses to be applied (if +ve, use LTP tables, if -ve, use LTD tables)
            jump_table     : The original jump table dict that was initialized
            self.min_conductance : Min. value of G as per the jump_table for G (LTP)
            self.max_conductance : Max. value of G as per the jump_table for G (LTD)
        '''
        numPulse = int(numPulse)
        if (numPulse > 0):
            # Apply LTP pulses
            # in the G jt, find the index of the value closest to G, call it near_G
            for _ in range(0, numPulse):
                near_G = find_nearest(self.set_G, G)
                prob = random.random()
                # in the CDF jt's near_G row, find the index of the value closest to prob, call it cdf (assumes CDF Array = G rows * dG cols)
                cdf = find_nearest(self.set_cdf[int(near_G)], prob)
                # finally, use the cdf index value as the appropriate dG to be added to the original conductance
                G = G + self.set_dG[int(cdf)]
                if (G > self.max_conductance):
                    G = self.max_conductance
                elif (G < self.min_conductance):
                    G = self.min_conductance
        elif (numPulse < 0):
            # Apply LTD pulses
            for _ in range(0, -1*numPulse):
                near_G = find_nearest(self.reset_G, G)
                prob = random.random()
                cdf = find_nearest(self.reset_cdf[int(near_G)], prob)
                G = G + self.reset_dG[int(cdf)]
                if (G > self.max_conductance):
                    G = self.max_conductance
                elif (G < self.min_conductance):
                    G = self.min_conductance
        return G