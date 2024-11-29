from xbtorch import get_xbtorch_param
import torch
import torch.nn as nn

def xbtorch_layer(cls):
    original_init = cls.__init__
    original_forward = cls.forward

    def xbtorch_init(self, *args, **kwargs):
        if (not get_xbtorch_param('initialized')): raise RuntimeError('XBTorch needs to be initialized, please refer to API for instructions.')
        original_init(self, *args, **kwargs)

        # for decomposition; currently not supported for multi-weight cells such as LSTM
        # TODO: Move this to lib's __init__ and do an in-based check
        if (hasattr(self, 'weight')):
            self.weight.input, self.weight.delta = None, None
            def backward_hook(module, grad_input, grad_output):
                self.weight.delta = grad_output[0]
            self.register_full_backward_hook(backward_hook)
        else:
            print(f'Multi-weight cell {type(self)} is currently not supported with decomposition algorithms.')

        # deployment (inference accleration)
        self._xb_inference = False
        self.inference_accelerator = get_xbtorch_param('inference_accelerator')

    def xbtorch_forward(self, input):
        # TODO: Add support for non-linear layers
        if (self._xb_inference):
            
            if (not hasattr(self, '_array_mappings')):
                raise ValueError("Array mappings are not present, likely an issue during initialization.")

            gnorm_scale = 1.0
            if (not self.inference_accelerator): raise ValueError('XB inference called without proper initialization of an accelerator profile.')

            # ternary mapping scheme
            v_read = self.inference_accelerator.v_read
            g_norm = self.inference_accelerator.g_max - self.inference_accelerator.g_min

            sw_weight = self.weight.data

            gamma = torch.unique(sw_weight)[-1] # WAGE quantization learns matrices [-gamma, 0, gamma], and so it's important to scale either G matrices or input voltage vector
                
            # convert inputs to voltages, then quantize to DAC-based precision
            input_voltages = input * v_read * gamma
            input_voltages = self.inference_accelerator.DAC_quantize(input_voltages)

            pos_idxs = self._array_mappings['Gpos']
            neg_idxs = self._array_mappings['Gneg']
        
            # pass to the crossbar, perform VMM, averaging/summing over multiple instances
            pos_outputs = []
            neg_outputs = []

            # Todo: can be possibly batched and made faster by using torch.bmm
            for pos_idx in pos_idxs:
                gpos = self.inference_accelerator.read_chip(pos_idx[0], sw_weight.shape[0], pos_idx[1], sw_weight.shape[1])
                pos_outputs.append(input_voltages @ gpos.T)

            for neg_idx in neg_idxs:
                gneg = self.inference_accelerator.read_chip(neg_idx[0], sw_weight.shape[0], neg_idx[1], sw_weight.shape[1])
                neg_outputs.append(input_voltages @ gneg.T)

            # Convert the list of tensors to a single tensor
            pos_outputs = torch.stack(pos_outputs)
            neg_outputs = torch.stack(neg_outputs)

            # for MAO, this has to be sum, for regular mapping, this will be average
            if (self._array_mappings['output_polling_mode'] == 'avg'):
                output = torch.mean(pos_outputs, dim=0) - torch.mean(neg_outputs, dim=0)
            elif (self._array_mappings['output_polling_mode'] == 'sum'):
                output = torch.sum(pos_outputs, dim=0) - torch.sum(neg_outputs, dim=0)
            elif (self._array_mappings['output_polling_mode'] == 'reduced_avg'):
                output = torch.sum(pos_outputs * self._array_mappings['maskpos'], dim=0) / self._array_mappings['alpha'] - torch.sum(neg_outputs * self._array_mappings['maskneg'], dim=0) / self._array_mappings['alpha']
            else:
                raise ValueError("output_polling_mode not implemented")
            # readback currents from ADC by simulated quantization again
            output = self.inference_accelerator.ADC_quantize(output) # equivalent to optimizing the TIA potentiometer resistance. ADC quantization 
            output = output / (gnorm_scale * g_norm * v_read)

            if (self.bias is not None): output += self.bias.data
            return output
        else:
            if (hasattr(self, 'weight')): self.weight.input = input
            output = original_forward(self, input)
            return output

    cls.__init__ = xbtorch_init
    cls.forward = xbtorch_forward
    return cls

def xbtorch_optimizer(cls):
    original_init = cls.__init__
    original_step = cls.step

    def xbtorch_init(self, *args, **kwargs):
        if (not get_xbtorch_param('initialized')): raise RuntimeError('XBTorch needs to be initialized, please refer to API for instructions.')
        self.decomp_alg = get_xbtorch_param('decomposition_algorithm')
        self.device_type = get_xbtorch_param('device_type')
        self.weight_range = get_xbtorch_param('weight_range')
        self.wage_quantize = get_xbtorch_param('wage_quantize')
        if (self.wage_quantize): 
            self.wage_params = get_xbtorch_param('wage_params')

        if not isinstance(self, torch.optim.SGD) and not isinstance(self, torch.optim.Adam):
            raise NotImplementedError("XBTorch only supports SGD and Adam optimizers.")

        original_init(self, *args, **kwargs)

    def xbtorch_step(self):

        for group_idx, group in enumerate(self.param_groups):
            for param_idx, param in enumerate(group['params']):
                group_param_idx = (group_idx, param_idx)

                # gradient quantization
                if (self.wage_quantize): # wage
                    # quantize gradient
                    if (not hasattr(param, 'weight_acc')): raise RuntimeError('Wage quantization used but parameters not initialized correctly. Likely an issue with model patching.')

                    lr = self.param_groups[group_idx]['lr']
                
                    # if (param.grad is None): param.grad = torch.zeros_like(param) # avoid undefined errors for tensors not requiring grad (hidden states for example in RNNs)

                    param.grad.data = self.wage_params['quantizer_grad'](param.grad.data, lr).data
                    
                    # quantize momentum
                    # this is experimental
                    if isinstance(self, torch.optim.SGD):
                        if hasattr(self, 'state'):
                            for state in self.state.values():
                                if state['momentum_buffer'] is not None:
                                    # Apply the operation to the momentum buffer
                                    state['momentum_buffer'] = self.wage_params['quantizer_grad'](state['momentum_buffer'], lr)
                    elif isinstance(self, torch.optim.Adam):
                        for state in self.state.values():
                            if state['exp_avg'] is not None:
                                # Apply the operation to the first moment estimate
                                state['exp_avg'] = self.wage_params['quantizer_grad'](state['exp_avg'], lr)
                            if state['exp_avg_sq'] in state:
                                # Apply the operation to the second moment estimate
                                state['exp_avg_sq'] = self.wage_params['quantizer_grad'](state['exp_avg_sq'], lr)

                if (self.decomp_alg): # a decomposition algorithm has been specified
                    param.grad = self.decomp_alg.decompose(param.input, param.delta, param.grad, group_param_idx)

                if (self.device_type): # a device type has been specified, so we include device weight modeling
                    pulse = self.device_type.gradient_to_pulse(param.grad)
                    if (torch.max(abs(pulse)) == 0 and group_idx == 0 and param_idx == 0):
                        print('Gradient during device update is becoming zero. Consider increasing learning rate if this continues.')
                    elif (torch.max(abs(pulse)) > 100 and group_idx == 0 and param_idx == 0):
                        print('Gradient during device update is exploding. Consider decreasing learning rate if this continues.')
                    conductances = self.device_type.weight_to_conductance(param.data)
                    conductances = self.device_type.write(conductances, -1*pulse, group_param_idx)
                    new_weights = self.device_type.conductance_to_weight(conductances)
                    param.grad = param.data - new_weights

                if (self.wage_quantize and param.ndimension() > 1): # skip over biases
                    # WAGE accumulate weight in gradient precision
                    # assume no batch norm
                    w_acc = self.wage_params['grad_clip'](param.weight_acc)
                    w_acc -= param.grad.data
                    param.weight_acc = w_acc

                    new_weights = self.wage_params['quantizer_weight'](
                        param.weight_acc, param.weight_scale
                    )

                    param.grad = (param.data - new_weights) * (1/lr) # in order to maintain consistency with optimizer.step() 

        original_step(self) #  can be done inside loop to avoid another loop at end

        if (self.device_type and not self.wage_quantize): # sw clipping, only done when WAGE is not used
            for group_idx, group in enumerate(self.param_groups):
                for param_idx, param in enumerate(group['params']):
                    param.data = torch.clamp(param.data, self.weight_range[0], self.weight_range[1]) # clamp bw range
        
        return None

    cls.__init__ = xbtorch_init
    cls.step = xbtorch_step
    return cls

if __name__ == '__main__':
    pass