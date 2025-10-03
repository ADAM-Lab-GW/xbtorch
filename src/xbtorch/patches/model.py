"""
Decorators for patching PyTorch models to make them compatible
with XBTorch operations.
"""
from xbtorch import get_xbtorch_param

import xbtorch.quant.wage_init as wage_init

import torch.nn as nn
import xbtorch.nn as xbnn
import xbtorch

import torch

def xbtorch_model(original_model):
    # TODO: This should be used if a model was already created i.e. on an instance
    # Copies state dictionary as well
    if (not get_xbtorch_param('initialized')): raise RuntimeError('XBTorch needs to be initialized, please refer to API for instructions.')

    wage_quantize = get_xbtorch_param('wage_quantize')
    xb_inference_accelerator = get_xbtorch_param('inference_accelerator')
    original_model.xb_forward = False # declare this to be false, xb_eval() has to be explicitly called
    
    # detect the device from the original model
    try:
        device = next(original_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")  # fallback if model has no parameters
    
    if (wage_quantize):
        wage_params = get_xbtorch_param('wage_params')
        quantizer_act_error = wage_params['quantizer_act_error']
        wl_activation = wage_params['wl_activation']
        wl_error = wage_params['wl_error']
        wl_weight = wage_params['wl_weight']

    if (not hasattr(original_model, 'model')): raise RuntimeError('Unable to find module list for patching, see network training example for correct patching workflow.')

    new_model = []
    if (wage_quantize): 
        new_model.append(quantizer_act_error(wl_activation, -1))

    for module in original_model.model:
        # TODO: A cleaner way to implement this could be to specify regex patterns for layers to be patched, or just specify them as a list
        # This should simplify model definitions, as well as patched re-creations here
        if (type(module) in xbtorch.layer_types):
            if (type(module)) == nn.Linear:
                args = (module.in_features, module.out_features, module.bias is not None)
                xbnn_layer = xbnn.Linear(*args)
            elif (type(module)) == nn.Conv2d:
                args = (module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias is not None, module.padding_mode)
                xbnn_layer = xbnn.Conv2d(*args)

            elif (type(module)) == nn.RNN:
                args = (module.input_size, module.hidden_size, module.num_layers, module.nonlinearity, module.bias, module.batch_first, module.dropout, module.bidirectional)
                xbnn_layer = xbnn.RNN(*args)

            elif (type(module)) == nn.LSTM:
                args = (module.input_size, module.hidden_size, module.num_layers, module.bias, module.batch_first, module.dropout, module.bidirectional, module.proj_size)
                xbnn_layer = xbnn.LSTM(*args)
            else:
                raise ValueError(f"An xbtorch supported layer, {type(module)}, is missing an implementation.")

            # move to device before loading weights
            xbnn_layer = xbnn_layer.to(device)
            xbnn_layer.load_state_dict(module.state_dict())
            xbnn_layer._array_mappings = {} # we add this to make initialization easier later, since pre-trained weights would be loaded after patching
            new_model.append(xbnn_layer)

        # activations
        elif (type(module) in xbtorch.activation_types): 
            new_model.append(module)
            if (wage_quantize): new_model.append(quantizer_act_error(wl_activation, wl_error))

        elif (type(module) in xbtorch.misc_types): 
            new_model.append(module)

        # else copy unpatched module, but notify user
        else:
            print(f'XBPatching for module {module} is not defined, using as is.')
            new_model.append(module)
            # exit()

    if (wage_quantize and new_model[-1] not in xbtorch.activation_types): 
        # add act/error quantizer if the last module is an activation
        new_model.append(quantizer_act_error(-1, wl_error))

    original_model.model = nn.Sequential(*new_model)

    if (wage_quantize):
        # wage parameters
        original_model.weight_scale = {}
        original_model.weight_acc = {}
        for name, param in original_model.named_parameters():
            if ("weight" in name): wage_init.wage_init_(param, wl_weight, factor=1.0)
            param.weight_acc = param.data

    # print('Patched XBTorch Model', original_model.model)

    if (xb_inference_accelerator):
        # if initialization included an inference accelerator
        def toggle(enable=True):
            original_model.xb_forward = enable
            for module in original_model.model:
                module._xb_inference = enable

        def initialize_array_mappings(output_polling_mode='avg', existing_mappings=[], additional_args={}):
            # existing_mappings can be used as a reference to avoid conflicting mappings across unique models on the xb (primary use case: committee machines)
            # reset/initialize mappings of this layer on the simulated crossbar
            original_model._array_mappings_all = existing_mappings

            if (output_polling_mode not in ['avg', 'sum', 'reduced_avg']):
                raise ValueError("Invalid output polling mode provided. Valid options are `avg` and `sum`.")

            for module in original_model.model:
                if (hasattr(module, '_array_mappings')):
                    sw_weight = module.weight.data # get weight data for this particular module (can be 1 or more weight matrices)
                    # get indices where the conductance matrices will be mapped on the simulated xbar
                    # xb_mapping_schemes = ['random', 'layer_ensemble'] etc.
                    pos_idxs = xb_inference_accelerator.xb_mapping_scheme(accelerator=xb_inference_accelerator, 
                                                                          layer_shape=sw_weight.shape, 
                                                                          current_mappings=original_model._array_mappings_all)
                    
                    neg_idxs = xb_inference_accelerator.xb_mapping_scheme(accelerator=xb_inference_accelerator, 
                                                                          layer_shape=sw_weight.shape,
                                                                          current_mappings=original_model._array_mappings_all)

                    # map the sw_weight matrix to the simulated array as device conductances at indices extracted above
                    # internally handles conversion of the sw_weight matrix to conductance matrices
                    # weight_encoding_schemes = ['regular', 'MAO']
                    maskspos, masksneg = xb_inference_accelerator.map_weights_to_array(sw_weight, pos_idxs=pos_idxs, neg_idxs=neg_idxs, additional_args=additional_args)
                    
                    # Attach information to layer for output gathering for inference
                    module._array_mappings['Gpos'] = pos_idxs
                    module._array_mappings['Gneg'] = neg_idxs

                    module._array_mappings['output_polling_mode'] = output_polling_mode

                    if (maskspos is not None and masksneg is not None):
                        module._array_mappings['maskpos'] = maskspos.unsqueeze(1) # unsqueeze to allow broadcasting on batch dimension
                        module._array_mappings['maskneg'] = masksneg.unsqueeze(1)
                        module._array_mappings['alpha'] = additional_args['alpha']

        def get_array_mappings():
            # If new encoding methods are added, this would have to be adjusted.
            output_dict = []
            for module in original_model.model:
                if (hasattr(module, '_array_mappings')):
                    output_dict.append({"Gpos": module._array_mappings["Gpos"], "Gneg": module._array_mappings["Gneg"],})
            return output_dict

        # attach new methods to the model
        original_model.xb_eval = toggle
        original_model.initialize_array_mappings = initialize_array_mappings
        original_model.get_array_mappings = get_array_mappings

    return original_model