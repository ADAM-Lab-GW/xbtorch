from xbtorch import get_xbtorch_param

import xbtorch.quant.wage_init as wage_init

import torch.nn as nn
import xbtorch.nn as xbnn
import xbtorch

def xbtorch_model(original_model):
    # TODO: This should be used if a model was already created i.e. on an instance
    # Copies state dictionary as well
    if (not get_xbtorch_param('initialized')): raise RuntimeError('XBTorch needs to be initialized, please refer to API for instructions.')

    wage_quantize = get_xbtorch_param('wage_quantize')
    if (wage_quantize):
        wage_params = get_xbtorch_param('wage_params')
        quantizer_act_error = wage_params['quantizer_act_error']
        wl_activation = wage_params['wl_activation']
        wl_error = wage_params['wl_error']
        wl_weight = wage_params['wl_weight']

    if (not hasattr(original_model, 'model')): raise RuntimeError('Unable to find module list for patching, see network training example for correct patching workflow.')

    new_model = []
    if (wage_quantize): new_model.append(quantizer_act_error(wl_activation, -1))

    for module in original_model.model:
        # layers
        if (type(module) in xbtorch.layer_types): 
            args = (module.in_features, module.out_features, module.bias is not None)
            xbnn_layer = xbnn.Linear(*args)
            xbnn_layer.load_state_dict(module.state_dict())
            new_model.append(xbnn_layer)
        # activations
        elif (type(module) in xbtorch.activation_types): 
            new_model.append(module)
            if (wage_quantize): new_model.append(quantizer_act_error(wl_activation, wl_error))

        # TODO: Copy state dictionary

    if (wage_quantize): new_model.append(quantizer_act_error(-1, wl_error))

    original_model.model = nn.Sequential(*new_model)

    if (wage_quantize):
        # wage parameters
        original_model.weight_scale = {}
        original_model.weight_acc = {}
        for name, param in original_model.named_parameters():
            assert "weight" in name
            wage_init.wage_init_(param, wl_weight, factor=1.0)
            param.weight_acc = param.data

    print('Patched XBTorch Model', original_model.model)

    return original_model