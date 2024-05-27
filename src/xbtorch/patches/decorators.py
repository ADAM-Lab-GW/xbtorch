from xbtorch import get_xbtorch_param
import torch

def xbtorch_layer(cls):
    original_init = cls.__init__
    original_forward = cls.forward

    def xbtorch_init(self, *args, **kwargs):
        if (not get_xbtorch_param('initialized')): raise RuntimeError('XBTorch needs to be initialized, please refer to API for instructions.')
        original_init(self, *args, **kwargs)
        self.weight.input, self.weight.delta = None, None
        def backward_hook(module, grad_input, grad_output):
            self.weight.delta = grad_output[0]
        self.register_full_backward_hook(backward_hook)

    def xbtorch_forward(self, input):
        self.weight.input = input
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
        if (self.wage_quantize): self.wage_params = get_xbtorch_param('wage_params')
        original_init(self, *args, **kwargs)

    def xbtorch_step(self):

        for group_idx, group in enumerate(self.param_groups):
            for param_idx, param in enumerate(group['params']):
                group_param_idx = (group_idx, param_idx)

                # gradient quantization
                if (self.wage_quantize): # wage
                    lr = self.param_groups[group_idx]['lr']
                    param.grad.data = self.wage_params['quantizer_grad'](param.grad.data, lr).data

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

                if (self.wage_quantize):
                    # WAGE accumulate weight in gradient precision
                    # assume no batch norm
                    if (not hasattr(param, 'weight_acc')): raise RuntimeError('Wage quantization used but parameters not initialized correctly. Likely an issue with model patching.')
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