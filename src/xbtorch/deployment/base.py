"""
Hardware accelerator simulation module.

Defines the abstract base class `GenericAccelerator` and several
concrete accelerator models (`SimpleFixedPoint`, `Daffodil`) for
simulating memristive crossbar arrays in XBTorch.

Features include:

- DAC and ADC quantization (fixed-point or board-specific)
- Noise modeling (read/write)
- Stuck-at defect simulation
- Input encoding schemes
- Weight encoding and mapping schemes
- Array visualization and utility functions

This module provides a unified interface for simulating hardware
effects on neural network training and inference.
"""

import abc
import torch
from qtorch.quant import fixed_point_quantize
import numpy as np

from xbtorch.deployment.mapping import map_random
from xbtorch.deployment.weight_encoding import encode_simple_binary, encode_LEA1, encode_LEA2

ACCELERATOR_REGISTRY = {}

def register_accelerator(name: str):
    """Decorator to register a custom layer under a string name."""
    def decorator(cls):
        ACCELERATOR_REGISTRY[name] = cls
        return cls
    return decorator

@register_accelerator("Generic")
class GenericAccelerator(metaclass=abc.ABCMeta):
    """
    Abstract base class for hardware accelerator models in XBTorch.

    This class simulates memristive crossbar arrays, incorporating
    hardware non-idealities such as stuck devices, read/write noise,
    limited precision DAC/ADC, input encoding schemes, and weight encoding/mapping schemes.
    Provided subclasses implement specific quantization methods for DAC and ADC. 

    Parameters
    ----------
    g_min : float
        Minimum device conductance (Siemens).
    g_max : float
        Maximum device conductance (Siemens).
    v_read : float
        Voltage used for read operations.
    read_noise : float
        Amplitude of uniform read noise applied during chip readout.
    write_noise : float
        Standard deviation of Gaussian noise applied during weight writes. Simulates device programming error.
    stateful: bool, optional
        In stateful mode, a physical representation of the entire crossbar is maintained, and weights are mapped to these limited devices.
        In stateless mode, weights are mapped and VMM is performed on the fly. This is more memory-efficient. Essentially behaves like an infinite size stateful crossbar.
    xb_size : tuple of int, optional
        Dimensions of the crossbar array (columns, rows). Default: (2500, 2500). Utilized only when stateful is True.
    stuck_percentage : float, optional
        Fraction of devices randomly stuck at high or low values. Default: 0.0.
    stuck_mode : {"ideal", "real"}, optional
        Mode for stuck-at defect modeling:
        - "ideal": stuck devices are fixed at g_min or g_max.
        - "real": stuck devices are fixed at predefined realistic values.
    input_encoding_scheme : {"instant", "linear"}, optional
        Mode for stuck-at defect modeling:
        - "instant": Instantaneous voltage-amplitude encoding of inputs.
        - "linear": Linear, bit-sliced encoding of inputs.
    weight_encoding_scheme : callable, optional
        Function used to encode weights into conductance matrices.
        Default: :func:`encode_simple_binary`.
    xb_mapping_scheme : callable, optional
        Function for mapping weights to crossbar positions.
        Default: :func:`map_random`.
    retention_time : float, optional
        elapsed deployment time.
        Default: 1.0.
    drift_coefficient : float, optional
        nominal ν.
        Default: 0.0.
    drift_t0 : float, optional
        reference time.
        Default: 1.0.
    drift_variation : float, optional
        device-to-device spread in ν.
        Default: 0.0.

    device : str, optional
        PyTorch device for simulation (e.g., "cpu" or "cuda").

    Attributes
    ----------
    _chip : torch.Tensor
        Simulated crossbar array storing device conductances.
    defect_map : tuple
        A tuple of (indices, values) representing defective devices.
    name : str
        Descriptive string identifying array dimensions and defect rate.

    Notes
    -----
    - Weight mapping uses both encoding (binary, LEA1, LEA2, etc.) and
      placement (e.g., random mapping).
    - Noise models include Gaussian for write noise and uniform for
      read noise.
    """
    
    def __init__(self, 
                 g_min, 
                 g_max, 
                 v_read, 
                 read_noise, 
                 write_noise, 
                 stateful=True,
                 xb_size=(2500, 2500), 
                 stuck_percentage=0.0, 
                 stuck_mode='real', 
                 input_encoding_scheme='instant',
                 weight_encoding_scheme=encode_simple_binary, 
                 xb_mapping_scheme=map_random,
                 retention_time=1.0,
                 drift_coefficient=0.0,
                 drift_t0=1.0,
                 drift_variation=0.0,
                 device="cpu"):

        self.read_noise = read_noise
        self.write_noise = write_noise
        self.g_min = g_min
        self.g_max = g_max
        self.v_read = v_read
        self.input_encoding_scheme = input_encoding_scheme
        self.weight_encoding_scheme = weight_encoding_scheme
        self.xb_mapping_scheme = xb_mapping_scheme
        self.stuck_percentage = stuck_percentage
        self.stateful = stateful

        self.retention_time = retention_time
        self.drift_coefficient = drift_coefficient
        self.drift_t0 = drift_t0
        self.drift_variation = drift_variation

        if (self.stuck_percentage > 0 and not self.stateful):
            raise ValueError("Stuck devices can not be simulated without a stateful representation of a crossbar. See examples for usage.")
    
        self.columns, self.rows = xb_size if stateful else (-1, -1)
        # self.stuck_low = 0
        # self.stuck_high = self.g_max * 2
        self.stuck_mode = stuck_mode
        if stuck_mode == 'ideal':
            self.stuck_low = g_min
            self.stuck_high = g_max
        elif stuck_mode == 'real':
            # could be passed as args
            self.stuck_low = 10
            self.stuck_high = 500
        else:
            raise ValueError(f"Stuck mode {stuck_mode} not implemented")

        if input_encoding_scheme not in ["instant", "linear"]:
            raise ValueError(f"IO Encoding mode {input_encoding_scheme} not implemented")

        # Create defect map
        # TODO: Separate out defect maps;

        self.name = f'stateful_{self.stateful}_cols_{self.columns}_row_{self.rows}_stuck_{self.stuck_percentage}'

        self.device = device

        if (self.stateful):
            self.initialize_chip()

    def initialize_chip(self):
        """
        Initialize the simulated chip state, assuming stateful mode. 
        If stateless, defect maps will be patched on dynamically during operation.

        - Fills the array with uninitialized values (-1).
        - Generates a defect map based on the specified stuck percentage.
        """
        if (self.stateful):
            self._chip = torch.ones((self.columns, self.rows)).to(self.device) * -1 # uninitialized devices
            self.defect_map = self.gen_defect_map(self.stuck_percentage) # defect map is a paired list of (defective indices, defective conductance states)
            self._chip[self.defect_map[0]] = self.defect_map[1]

    def get_xb_size(self):
        """
        Retrieve the size (rows, columns) of the simulated crossbar.
        """
        return (self.columns, self.rows)

    def apply_conductance_drift(self, G):
        """
        Apply retention-induced conductance drift.

        Parameters
        ----------
        G

        Returns
        -------
        torch.Tensor
            Subarray with applied drift.

        """

        if self.drift_coefficient == 0:
            return G

        if self.drift_variation > 0:
            # device-to-device variation
            nu = torch.normal(
                mean=self.drift_coefficient,
                std=self.drift_variation,
                size=G.shape,
                device=G.device
            )
        else:
            # deterministic global drift
            nu = self.drift_coefficient

        drift_factor = (self.retention_time / self.drift_t0) ** (-nu)

        G_drifted = G * drift_factor
        G_drifted = torch.clamp(G_drifted, self.g_min, self.g_max)

        return G_drifted

    def read_chip(self, row, n_rows, col, n_cols, fast_mode=True):
        """
        Read a subarray of the chip, optionally with read noise.

        This method requires the object to be in a stateful mode.
        If `self.stateful` is False, a RuntimeError is raised.

        Parameters
        ----------
        row : int
            Starting row index.
        n_rows : int
            Number of rows to read.
        col : int
            Starting column index.
        n_cols : int
            Number of columns to read.
        fast_mode : bool, optional
            If False, raises NotImplementedError. Default: True.

        Returns
        -------
        torch.Tensor
            Subarray with applied read noise (if configured).


        Raises
        ------
        RuntimeError
            If `self.stateful` is False.
        ValueError
            If `fast_mode` is False (not implemented yet).

        """

        if not self.stateful:
            raise RuntimeError("Cannot read chip when self.stateful is False.")

        subarray = self._chip[row:row+n_rows, col:col+n_cols]

        # conductance drift
        subarray = self.apply_conductance_drift(subarray)

        # read noise
        noise = torch.empty_like(subarray).uniform_(-self.read_noise, self.read_noise)
        if (not fast_mode): raise ValueError("Not implemented")
        if (self.read_noise > 0): subarray = subarray + noise
        return torch.clamp(subarray, self.g_min, self.g_max)

    def read_chip_stateless(self, subarray):
        """
        Read a subarray of the chip, optionally with read noise.

        This method requires the object to be in a stateful mode.
        If `self.stateful` is False, a RuntimeError is raised.

        Parameters
        ----------
        G

        Returns
        -------
        torch.Tensor
            Subarray with applied read noise (if configured).


        Raises
        ------
        RuntimeError
            If `self.stateful` is False.
        ValueError
            If `fast_mode` is False (not implemented yet).

        """

        noise = torch.empty_like(subarray).uniform_(-self.read_noise, self.read_noise)
        subarray = self.apply_conductance_drift(subarray)
        if (self.read_noise > 0): subarray = subarray + noise
        return torch.clamp(subarray, self.g_min, self.g_max)

    def gen_defect_map(self, stuck_percentage):
        """
        Generate a defect map for the chip.

        Parameters
        ----------
        stuck_percentage : float
            Fraction of devices that are stuck.

        Returns
        -------
        tuple
            (indices, values) where:
            - indices are tensor indices of defective devices,
            - values are their fixed conductances (stuck_high or stuck_low).
        """
        if (not self.stateful):
            return
        
        num_elements = int(stuck_percentage * self._chip.numel())
        defect_indices = np.unravel_index(
            np.random.choice(self._chip.shape[0] * self._chip.shape[1], num_elements, replace=False), (self._chip.shape[0], self._chip.shape[1])
        )
        defect_values = torch.randint(0, 2, (num_elements,), dtype=torch.float32)
        defect_values[defect_values == 0] = self.stuck_low
        defect_values[defect_values == 1] = self.stuck_high
        return defect_indices, defect_values.to(self.device)

    def map_weights_to_array_stateless(self, sw_weight):

        if (self.stateful):
            return

        encoded_return =  self.weight_encoding_scheme(self, sw_weight)
        Gposs, Gnegs = encoded_return[0], encoded_return[1]
        sw_weight_shape = sw_weight.shape

        # write noise
        for i, Gpos in enumerate(Gposs):
            if (self.write_noise > 0):
                noise = torch.randn_like(Gposs[i]) * self.write_noise + 0.0 # 0 mean
                Gposs[i] = torch.clamp(Gposs[i] + noise, self.g_min, self.g_max)

        for i, Gneg in enumerate(Gnegs):
            if (self.write_noise > 0):
                noise = torch.randn_like(Gnegs[i]) * self.write_noise + 0.0 # 0 mean
                Gnegs[i] = torch.clamp(Gnegs[i] + noise, self.g_min, self.g_max)

        return Gposs, Gnegs

    def map_weights_to_array(self, sw_weight, pos_idxs=[], neg_idxs=[], additional_args={}):        
        """
        Map software weights onto the hardware array.

        Parameters
        ----------
        sw_weight : torch.Tensor
            Software weight matrix to map.
        pos_idxs : list of tuple, optional
            Starting indices for positive weight encodings.
        neg_idxs : list of tuple, optional
            Starting indices for negative weight encodings.
        additional_args : dict, optional
            Additional keyword arguments for the encoding scheme.

        Returns
        -------
        tuple
            (masksposs, masksnegs) masks used for layer ensemble
            averaging schemes. Returns (None, None) otherwise.

        Notes
        -----
        - Adds Gaussian write noise if configured.
        - Defect map is reapplied to enforce stuck devices.
        """

        if (not self.stateful):
            return

        encoded_return =  self.weight_encoding_scheme(self, sw_weight, pos_idxs=pos_idxs, neg_idxs=neg_idxs, additional_args=additional_args)
        Gposs, Gnegs = encoded_return[0], encoded_return[1]
        sw_weight_shape = sw_weight.shape

        for i, pos_idx in enumerate(pos_idxs):
            if (self.write_noise > 0):
                noise = torch.randn_like(Gposs[i]) * self.write_noise + 0.0 # 0 mean
                Gposs[i] = torch.clamp(Gposs[i] + noise, self.g_min, self.g_max)

            self._chip[pos_idx[0]:pos_idx[0]+sw_weight_shape[0], 
                    pos_idx[1]:pos_idx[1]+sw_weight_shape[1]] = Gposs[i]

        for i, neg_idx in enumerate(neg_idxs):
            if (self.write_noise > 0):
                noise = torch.randn_like(Gnegs[i]) * self.write_noise + 0.0 # 0 mean
                Gnegs[i] = torch.clamp(Gnegs[i] + noise, self.g_min, self.g_max)

            self._chip[neg_idx[0]:neg_idx[0]+sw_weight_shape[0], 
                    neg_idx[1]:neg_idx[1]+sw_weight_shape[1]] = Gnegs[i]

        # Add back defect map information in case the outer method attempted to do an illegal assignment
        self._chip[self.defect_map[0]] = self.defect_map[1]

        masksposs, masksnegs = None, None
        if (self.weight_encoding_scheme == encode_LEA1 or self.weight_encoding_scheme == encode_LEA2):
            masksposs, masksnegs = encoded_return[2], encoded_return[3]

        return masksposs, masksnegs

    @abc.abstractmethod
    def DAC_quantize(self, vector):
        """
        Abstract method to quantize input voltages via DAC.

        Parameters
        ----------
        vector : torch.Tensor
            Full-precision input voltage vector.

        Returns
        -------
        torch.Tensor
            Quantized voltage vector.
        """
        pass

    @abc.abstractmethod
    def ADC_quantize(self, vector):
        """
        Abstract method to quantize output currents via ADC.

        Parameters
        ----------
        vector : torch.Tensor
            Full-precision current vector.

        Returns
        -------
        torch.Tensor
            Quantized current vector.
        """
        pass
    
    def plot_array(self, x_start=None, x_count=None, y_start=None, y_count=None, title=None, show=False):
        """
        Visualize the conductance state of the array.

        Parameters
        ----------
        x_start, y_start : int, optional
            Starting indices of the subarray to plot.
        x_count, y_count : int, optional
            Dimensions of the subarray to plot.
        title : str, optional
            Title for the plot.
        show : bool, optional
            Display the plot at the end.

        Returns
        -------
        torch.Tensor
            The read subarray.
        """

        if (not self.stateful):
            return

        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.xlabel('Column #')
        plt.ylabel('Row #')

        if (x_start is None or x_count is None or y_start is None or y_count is None):
            # read the full array
            xedges = list(range(self._chip.shape[0]))
            yedges = list(range(self._chip.shape[1]))
            read_chip = self.read_chip(0, self._chip.shape[0], 0, self._chip.shape[1], fast_mode=True).cpu()
        else:
            # read only requested subset of the array
            xedges = list(range(x_count))
            yedges = list(range(y_count))
            read_chip = self.read_chip(x_start, x_count, y_start, y_count, fast_mode=True).cpu()
        surf = plt.pcolormesh(xedges, yedges, read_chip.T,  alpha=1, antialiased=True, linewidth=0.0, zorder=-1)
        surf.set_edgecolor('face')
        plt.axis('image')
        if (title):
            plt.title(title)
        if show: plt.show()
        return read_chip

@register_accelerator("SimpleFixedPoint")
class SimpleFixedPoint(GenericAccelerator):
    """
    Simple fixed-point accelerator model.

    This accelerator simulates fixed-point DAC and ADC quantization
    using the QTorch library. It models limited precision effects
    with configurable bitwidths.

    Parameters
    ----------
    adc_bits : int, optional (default=5)
        Number of bits for ADC quantization.
    dac_bits : int, optional (default=5)
        Number of bits for DAC quantization.
    **kwargs :
        See :class:`GenericAccelerator` for supported parameters.

    Notes
    -----

    - Quantization is symmetric, using QTorch’s
      :func:`fixed_point_quantize`.

    """

    def __init__(self, 
                 adc_bits=5, 
                 dac_bits=5, 
                 g_min=50, 
                 g_max=100, 
                 v_read=0.3, 
                 read_noise=0, 
                 stateful=True,
                 xb_size=(2500, 2500), 
                 write_noise=0, 
                 stuck_percentage=0.0, 
                 stuck_mode='real', 
                 input_encoding_scheme='instant',
                 xb_mapping_scheme=map_random, 
                 weight_encoding_scheme=encode_simple_binary,
                 retention_time=1.0,
                 drift_coefficient=0.0,
                 drift_t0=1.0,
                 drift_variation=0.0, 
                 device='cpu'):
                # TODO: Input encoding modes and output encoding modes should go here
        super().__init__(g_min, 
                         g_max, 
                         v_read, 
                         read_noise=read_noise, 
                         write_noise=write_noise, 
                         stateful=stateful, 
                         xb_size=xb_size, 
                         stuck_percentage=stuck_percentage,
                         stuck_mode=stuck_mode,
                         input_encoding_scheme=input_encoding_scheme,
                         xb_mapping_scheme=xb_mapping_scheme,
                         weight_encoding_scheme=weight_encoding_scheme, 
                         retention_time=retention_time,
                         drift_coefficient=drift_coefficient,
                         drift_t0=drift_t0,
                         drift_variation=drift_variation,
                         device=device)
        self.adc_bits = adc_bits
        self.dac_bits = dac_bits

    def DAC_quantize(self, vector):
        """
        Quantize input voltage vector using fixed-point DAC.

        Parameters
        ----------
        vector : torch.Tensor
            Full-precision input voltage vector.

        Returns
        -------
        torch.Tensor
            Quantized voltage vector.
        """

        max_val = torch.max(torch.abs(vector))

        if self.input_encoding_scheme == "instant":
        
            return max_val * fixed_point_quantize(
                vector / max_val,
                wl=self.dac_bits,
                fl=self.dac_bits - 1,
                symmetric=True
            )

        elif self.input_encoding_scheme == "linear":

            normalized = torch.clamp(vector / max_val, -1.0, 1.0)

            levels = 2 ** self.dac_bits - 1
            quantized = torch.round(normalized * levels)

            slices = []
            sign = torch.sign(quantized)
            mag = torch.abs(quantized).long()

            for bit in range(self.dac_bits):
                bit_slice = ((mag >> bit) & 1).float()
                slices.append(sign * bit_slice * max_val)

            return slices

        else:
            raise ValueError("Unsupported encoding scheme")

    def ADC_quantize(self, vector):
        """
        Quantize output current vector using fixed-point ADC.

        Parameters
        ----------
        vector : torch.Tensor
            Full-precision current vector.

        Returns
        -------
        torch.Tensor
            Quantized current vector.
        """

        max_val = torch.max(torch.abs(vector))

        return max_val * fixed_point_quantize(
            vector / max_val,
            wl=self.adc_bits,
            fl=self.adc_bits - 1,
            symmetric=True
        )

        # TODO: Support for ADC per-slice is not implemented

@register_accelerator("Daffodil")
class Daffodil(GenericAccelerator):
    """
    Experimental Daffodil accelerator model.

    This accelerator simulates the Daffodil prototyping system
    developed by NIST/GW/WD, with detailed modeling of the on-board
    DAC/ADC behavior based on datasheets and experimental calibration.

    Parameters
    ----------
    **kwargs :
        See :class:`GenericAccelerator` for supported parameters.

    Notes
    -----
    - Models a 12-bit DAC (AD5391BSTZ-5) and an ADC with calibration.
    - Includes board-level parameters such as reference voltages,
      gains, and transimpedance amplifier settings.
    - Provides helper functions for DAC/ADC conversions.

    References
    ----------
    - NIST/GW/WD Daffodil Prototyping System:
      https://arxiv.org/abs/2404.15621
    - AD5391BSTZ-5 Datasheet:
      https://www.analog.com/media/en/technical-documentation/data-sheets/AD5390_5391_5392.pdf
    
    """

    def __init__(self,
                 g_min=50,
                 g_max=100,
                 v_read=0.3,
                 read_noise=10,
                 write_noise=10,
                 stuck_percentage=0.0,
                 stuck_mode='real',
                 input_encoding_scheme='instant',
                 xb_mapping_scheme=map_random,
                 retention_time=1.0,
                 drift_coefficient=0.0,
                 drift_t0=1.0,
                 drift_variation=0.0, 
                 device='cpu'):
        super().__init__(g_min,
                         g_max,
                         v_read,
                         read_noise,
                         write_noise,
                         stuck_percentage,
                         stuck_mode,
                         input_encoding_scheme,
                         xb_mapping_scheme,
                         retention_time=retention_time,
                         drift_coefficient=drift_coefficient,
                         drift_t0=drift_t0,
                         drift_variation=drift_variation,
                         device=device)

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

        print("The Daffodil accelerator is currently experimental. Use with caution.")

    # Helper methods adapted from NIST/GW/WD's Daffodil prototyping system
    # https://arxiv.org/abs/2404.15621
    def dac_calcvout(self, x1):
        """
        Compute DAC analog output voltage for a digital register input.

        Parameters
        ----------
        x1 : torch.Tensor
            Register values (0 <= x1 <= 2^12).

        Returns
        -------
        torch.Tensor
            Corresponding DAC output voltages.
        """
        if torch.any(x1 < 0) or torch.any(x1  > self.dac_max_prec): raise ValueError("DAC x1 out of range")
        x2 = ((self.dac_gain_mode+2) / self.dac_max_prec)*x1 + (self.dac_offset) # digital input transfer function
        return 2 * self.dac_vref * x2/self.dac_max_prec 

    def dac_invertvout(self, v):
        """
        Compute nearest DAC register value for a given output voltage.

        Parameters
        ----------
        v : torch.Tensor
            Desired DAC output voltage.

        Returns
        -------
        torch.Tensor
            Nearest integer DAC register value.
        """
        reg = torch.round(2**(self.dac_n-1)*((self.dac_max_prec)*(v)-2* self.dac_offset*self.dac_vref)/(2+self.dac_gain_mode)/self.dac_vref)
        return torch.clamp(reg, 0, self.dac_max_prec - 1)

    def DAC_quantize(self, voltage):
        """
        Quantize input voltage via the DAC model.

        Parameters
        ----------
        voltage : torch.Tensor
            Full-precision voltage input.

        Returns
        -------
        torch.Tensor
            Quantized DAC output voltage.
        """
        x1 = self.dac_invertvout(voltage)
        vout = self.dac_calcvout(x1)
        return vout
    
    def adc_predict_voltage(self, registervalue): #this takes an ADC value and tells you how much voltage you should have gotten
        """
        Predict analog voltage from ADC register value.

        Parameters
        ----------
        registervalue : torch.Tensor
            ADC register value.

        Returns
        -------
        torch.Tensor
            Predicted analog voltage.
        """
        return (1+self.adc_gain)*self.adc_vref*registervalue/4096

    def adc_invert_voltage(self, value):#, gain): #this takes a voltage and tells you the nearest ADC value that maps to it.
        """
        Invert analog voltage to nearest ADC register value.

        Parameters
        ----------
        value : torch.Tensor
            Input voltage.

        Returns
        -------
        torch.Tensor
            ADC register value.

        Raises
        ------
        ValueError
            If input voltage exceeds ADC register range.
        """
        registers = torch.round(4096*value/(self.adc_gain+1)/self.adc_vref)
        if torch.any(registers > 4096): 
            print(torch.max(value))
            raise ValueError("ADC Register overflow")
        return registers

    def ADC_quantize(self, current):
        """
        Quantize output current via the ADC model.

        This method simulates the full signal chain:

        - Converts crossbar current to voltage using a transimpedance
          amplifier model.
        - Clips voltages to the ADC’s operating range.
        - Maps voltages to ADC register values.
        - Converts registers back to voltages.
        - Reconstructs quantized currents.

        Parameters
        ----------
        current : torch.Tensor
            Full-precision current vector (from crossbar readout).

        Returns
        -------
        torch.Tensor
            Quantized current vector, reconstructed from ADC output.

        Notes
        -----
        - Transimpedance conversion uses board parameters
          (``board_vref``, ``board_vmax``, ``board_vground``,
          ``dpot_r``, ``currentscale``).
        - Simulated read noise can be optionally added for realism,
          but is currently disabled in this implementation.
        """
        # assumption: voltages applied to columns, currents read from rows of the crossbar
        # Transimpedance stage
        transimpedance_output = self.board_vref + (current/self.currentscale)*self.dpot_r
        transimpedance_output = torch.clamp(transimpedance_output, 0.0, 
                                            self.board_vmax - self.board_vground)
        
        # Map voltage to ADC register and back
        adc_registers = self.adc_invert_voltage(transimpedance_output)
        vout = self.adc_predict_voltage(adc_registers)

        # Reconstruct current from voltage
        currents = ((vout-self.board_vref)/self.dpot_r) * self.currentscale

        # Optional noise model (currently disabled)
        # noise = torch.empty_like(currents).uniform_(-self.read_noise, self.read_noise)

        # currents = currents + noise

        return currents