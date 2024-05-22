from .. import get_xbtorch_param

def alter_layer(cls):
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

def alter_optimizer(cls):
    original_init = cls.__init__
    original_step = cls.step

    def xbtorch_init(self, *args, **kwargs):
        if (not get_xbtorch_param('initialized')): raise RuntimeError('XBTorch needs to be initialized, please refer to API for instructions.')
        self.decomp_alg = get_xbtorch_param('decomposition_algorithm')
        self.device_type = get_xbtorch_param('device_type')
        original_init(self, *args, **kwargs)

    def xbtorch_step(self):
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, param in enumerate(group['params']):
                group_param_idx = (group_idx, param_idx)

                if (self.decomp_alg): # a decomposition algorithm has been specified
                    param.grad = self.decomp_alg.decompose(param.input, param.delta, param.grad, group_param_idx)

                if (self.device_type): # a device type has been specified, so we include device weight modeling
                    pulse = self.device_type.gradient_to_pulse(param.grad)
                    conductances = self.device_type.weight_to_conductance(param.data)
                    conductances = self.device_type.write(conductances, -1*pulse, group_param_idx)
                    new_weights = self.device_type.conductance_to_weight(conductances)
                    param.grad = param.data - new_weights
    
        output = original_step(self)
        return output

    cls.__init__ = xbtorch_init
    cls.step = xbtorch_step
    return cls

if __name__ == '__main__':
    pass