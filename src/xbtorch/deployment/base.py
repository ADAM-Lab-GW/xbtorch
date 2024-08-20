import abc
import torch
from qtorch.quant import fixed_point_quantize
import numpy as np

from xbtorch.deployment.mapping import map_random
from xbtorch.deployment.encoding import encode_simple

class GenericAccelerator(metaclass=abc.ABCMeta):
    def __init__(self, g_min, g_max, v_read, read_noise, write_noise, stuck_percentage=0.0, stuck_mode='real', weight_encoding_scheme=encode_simple, xb_mapping_scheme=map_random):
        self.read_noise = read_noise
        self.write_noise = write_noise
        self.g_min = g_min
        self.g_max = g_max
        self.v_read = v_read
        self.weight_encoding_scheme = weight_encoding_scheme
        self.xb_mapping_scheme = xb_mapping_scheme
        self.stuck_percentage = stuck_percentage
    
        self.columns, self.rows = 2500, 2500
        # self.stuck_low = 0
        # self.stuck_high = self.g_max * 2
        self.stuck_mode = stuck_mode
        if stuck_mode == 'ideal':
            self.stuck_low = g_min
            self.stuck_high = g_max
        elif stuck_mode == 'real':
            # needs to be passed as args
            self.stuck_low = 10
            self.stuck_high = 500
        else:
            raise ValueError(f"Stuck mode {stuck_mode} not implemented")


        # Create defect map
        # TODO: Separate out defect maps

        self.name = f'cols_{self.columns}_row_{self.rows}_stuck_{self.stuck_percentage}'

        self.initialize_chip()

    def initialize_chip(self):
        self._chip = torch.ones((self.columns, self.rows)) * -1 # uninitialized devices
        self.defect_map = self.gen_defect_map(self.stuck_percentage) # defect map is a paired list of (defective indices, defective conductance states)
        self._chip[self.defect_map[0]] = self.defect_map[1]

    def read_chip(self, row, n_rows, col, n_cols, fast_mode=True):
        subarray = self._chip[row:row+n_rows, col:col+n_cols]
        noise = torch.empty_like(subarray).uniform_(-self.read_noise, self.read_noise)
        if (fast_mode): return subarray + noise
        else:
            # TODO: apply V_read row/column wise to the subarray, and extract G from it.
            raise ValueError("Not implemented")
            return 

    def gen_defect_map(self, stuck_percentage):

        # Define the number of random indices to generate
        num_elements = int(stuck_percentage * self._chip.numel())

        # Generate random indices
        defect_indices = np.unravel_index(
            np.random.choice(self._chip.shape[0] * self._chip.shape[1], num_elements, replace=False), (self._chip.shape[0], self._chip.shape[1])
        )

        # the devices will be randomly stuck high or low
        defect_values = torch.randint(0, 2, (num_elements,), dtype=torch.float32)

        defect_values[defect_values == 0] = self.stuck_low
        defect_values[defect_values == 1] = self.stuck_high

        return defect_indices, defect_values

    def map_weights_to_array(self, sw_weight, pos_idxs=[], neg_idxs=[]):        
        # Calculate Gpos and Gneg matrices for the given sw_weight matrix and map on to the array at specified idxs
        # the weight magnitude can technically be multiplied by the G matrices
        # alternately, input voltages can be scaled, which is better because it gives more control over G matrix values
        # Finally, map the Gpos and Gneg matrices to the chip, possibly more than once

        Gposs, Gnegs =  self.weight_encoding_scheme(self, sw_weight, pos_idxs=pos_idxs, neg_idxs=neg_idxs)
        sw_weight_shape = sw_weight.shape

        for i, pos_idx in enumerate(pos_idxs):
            if (self.write_noise > 0):
                # modelled as normally distributed
                noise = torch.randn_like(Gposs[i]) * self.write_noise + 0.0 # 0 mean
                Gposs[i] = Gposs[i] + noise

            self._chip[pos_idx[0]:pos_idx[0]+sw_weight_shape[0], pos_idx[1]:pos_idx[1]+sw_weight_shape[1]] = Gposs[i]

        for i, neg_idx in enumerate(neg_idxs):

            if (self.write_noise > 0):
                # modelled as normally distributed
                noise = torch.randn_like(Gnegs[i]) * self.write_noise + 0.0 # 0 mean
                Gnegs[i] = Gnegs[i] + noise

            self._chip[neg_idx[0]:neg_idx[0]+sw_weight_shape[0], neg_idx[1]:neg_idx[1]+sw_weight_shape[1]] = Gnegs[i]

        # Add back defect map information in case the outer method attempted to do an illegal assignment
        self._chip[self.defect_map[0]] = self.defect_map[1]

    @abc.abstractmethod
    def DAC_quantize(self, vector):
        # given a full-precision voltage vector, quantize based on DAC precisions
        pass

    @abc.abstractmethod
    def ADC_quantize(self, vector):
        # given a full-precision voltage vector, quantize based on DAC precisions
        pass
    
    def plot_array(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.xlabel('Column #')
        plt.ylabel('Row #')

        xedges = list(range(self._chip.shape[0]))
        yedges = list(range(self._chip.shape[1]))
        surf = plt.pcolormesh(xedges, yedges, self._chip.T,  alpha=1, antialiased=True, linewidth=0.0, zorder=-1)
        surf.set_edgecolor('face')
        plt.axis('image')
        plt.show()

class SimpleFixedPoint(GenericAccelerator):

    def __init__(self, adc_bits=5, dac_bits=5, g_min=50, g_max=100, v_read=0.3, read_noise=0, write_noise=0, stuck_percentage=0.0, stuck_mode='real', xb_mapping_scheme=map_random, weight_encoding_scheme='regular'):
        super().__init__(g_min, g_max, v_read, read_noise=read_noise, write_noise=write_noise, stuck_percentage=stuck_percentage, stuck_mode=stuck_mode, xb_mapping_scheme=xb_mapping_scheme, weight_encoding_scheme=weight_encoding_scheme)
        self.adc_bits = adc_bits
        self.dac_bits = dac_bits

    def DAC_quantize(self, vector):
        # given a full-precision voltage vector, quantize based on DAC precisions
        max_val = torch.max(vector)
        return max_val * fixed_point_quantize(vector / max_val, wl=self.dac_bits, fl=self.dac_bits-1, symmetric=True)

    def ADC_quantize(self, vector):
        # given a full-precision voltage vector, quantize based on DAC precisions
        max_val = torch.max(vector)
        return max_val * fixed_point_quantize(vector / max_val, wl=self.adc_bits, fl=self.adc_bits-1, symmetric=True)


class Daffodil(GenericAccelerator):
    def __init__(self, g_min=50, g_max=100, v_read=0.3, read_noise=10, write_noise=10, stuck_percentage=0.0, stuck_mode='real', xb_mapping_scheme=map_random):
        super().__init__(g_min, g_max, v_read, read_noise, write_noise, stuck_percentage, stuck_mode, xb_mapping_scheme)

        # Board level parameters, calibrated from hardware experiments
        # Can be overridenn based on further experimentation
        self.board_vref = 2.0
        self.board_vmax = 5
        self.board_vground = 1.7

        # simulate the 12-bit DAC part AD5391BSTZ-5 present on the Daffodil board
        # Device documentation available: https://www.analog.com/media/en/technical-documentation/data-sheets/AD5390_5391_5392.pdf
        self.dac_n = 12
        self.dac_max_prec = 2**self.dac_n
        self.dac_gain_mode = 4093 # m from the manual
        self.dac_offset = 0 # c - 2^(n-1) from the manual
        self.dac_vref = 2.5

        self.adc_gain = 0
        self.adc_vref = 2.5 # should be between 2 and 3 as per datasheet
        
        self.dpot_r = -0.5*10**3
        self.currentscale = 10**8

    # Helper methods adapted from NIST/GW/WD's Daffodil prototyping system
    # https://arxiv.org/abs/2404.15621
    def dac_calcvout(self, x1):
        # this calls the dac function to calculate the output from from a particular 12bit register value
        if torch.any(x1 < 0) or torch.any(x1  > self.dac_max_prec): raise ValueError("DAC x1 out of range")
        x2 = ((self.dac_gain_mode+2) / self.dac_max_prec)*x1 + (self.dac_offset) # digital input transfer function
        return 2 * self.dac_vref * x2/self.dac_max_prec 

    def dac_invertvout(self, v):
        reg = torch.round(2**(self.dac_n-1)*((self.dac_max_prec)*(v)-2* self.dac_offset*self.dac_vref)/(2+self.dac_gain_mode)/self.dac_vref)
        return torch.clamp(reg, 0, self.dac_max_prec - 1)

    def DAC_quantize(self, voltage):
        x1 = self.dac_invertvout(voltage)
        vout = self.dac_calcvout(x1)
        return vout
    
    def adc_predict_voltage(self, registervalue): #this takes an ADC value and tells you how much voltage you should have gotten
        return (1+self.adc_gain)*self.adc_vref*registervalue/4096

    def adc_invert_voltage(self, value):#, gain): #this takes a voltage and tells you the nearest ADC value that maps to it.
        registers = torch.round(4096*value/(self.adc_gain+1)/self.adc_vref)
        if torch.any(registers > 4096): 
            print(torch.max(value))
            raise ValueError("ADC Register overflow")
        return registers

    def ADC_quantize(self, current):
        # assumption: voltages applied to columns, currents read from rows of the crossbar
        transimpedance_output = self.board_vref + (current/self.currentscale)*self.dpot_r
        transimpedance_output = torch.clamp(transimpedance_output, 0.0, self.board_vmax - self.board_vground)
        adc_registers = self.adc_invert_voltage(transimpedance_output)
        vout = self.adc_predict_voltage(adc_registers)
        currents = ((vout-self.board_vref)/self.dpot_r) * self.currentscale

        # simulate read noise
        # Generate uniform noise in the specified range
        # noise = torch.empty_like(currents).uniform_(-self.read_noise, self.read_noise)

        # currents = currents + noise

        return currents