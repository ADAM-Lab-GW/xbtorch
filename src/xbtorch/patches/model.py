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
    xb_inference_accelerator = get_xbtorch_param('inference_accelerator')
    original_model.xb_forward = False # declare this to be false, xb_eval() has to be explicitly called
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
            xbnn_layer._array_mappings = {} # we add this to make initialization easier later, since pre-trained weights would be loaded after patching
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

    if (xb_inference_accelerator):
        # if initialization included an inference accelerator
        def toggle(enable=True):
            original_model.xb_forward = enable
            for module in original_model.model:
                module._xb_inference = True

        def initialize_array_mappings():
            original_model._array_mappings_all = []
            for module in original_model.model:
                if (hasattr(module, '_array_mappings')):
                    sw_weight = module.weight.data

                    pos_idxs, neg_idxs = xb_inference_accelerator.get_map_idxs(chip_shape=xb_inference_accelerator.chip.shape, 
                                                                                layer_shape=sw_weight.shape, 
                                                                                current_mappings=original_model._array_mappings_all)

                    xb_inference_accelerator.map_weights_to_array(sw_weight, pos_idxs=pos_idxs, neg_idxs=neg_idxs)
                    
                    module._array_mappings['Gpos'] = pos_idxs
                    module._array_mappings['Gneg'] = neg_idxs

        # attach new methods to the model
        original_model.xb_eval = toggle
        original_model.initialize_array_mappings = initialize_array_mappings

    return original_model