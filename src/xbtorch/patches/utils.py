import xbtorch.quant.wage_init as wage_init
import torch.nn as nn
import xbtorch.nn as xbnn
import xbtorch
import transformers

from xbtorch import get_xbtorch_param

def replace_all_layers_stateless(model: transformers.models, 
                              exclude: list=[]) -> None:
    """
        Recursively replace all modules within a model with XBTorch modules for stateless operation. 
    """
    for name, module in model.named_children():
        
        # this is not a nested exclusion
        if exclude and name in exclude:
            continue
        
        # TODO: could be a simple map between nn vs. xbnn modules, removing the elifs
        if isinstance(module, nn.Linear):
            args = (module.in_features, module.out_features, module.bias is not None)
            xbnn_layer = xbnn.Linear(*args).to(next(module.parameters()).device)
            xbnn_layer.load_state_dict(module.state_dict())
            xbnn_layer._xb_inference = False # add attribute for xb evaluation for proper toggling later
            setattr(model, name, xbnn_layer)
        elif (type(module)) == nn.Conv2d:
            args = (module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias is not None, module.padding_mode)
            xbnn_layer = xbnn.Conv2d(*args)
            xbnn_layer.load_state_dict(module.state_dict())
            xbnn_layer._xb_inference = False # add attribute for xb evaluation for proper toggling later
            setattr(model, name, xbnn_layer)
        elif (type(module)) == nn.RNN:
            args = (module.input_size, module.hidden_size, module.num_layers, module.nonlinearity, module.bias, module.batch_first, module.dropout, module.bidirectional)
            xbnn_layer = xbnn.RNN(*args)
            xbnn_layer.load_state_dict(module.state_dict())
            xbnn_layer._xb_inference = False # add attribute for xb evaluation for proper toggling later
            setattr(model, name, xbnn_layer)
        elif (type(module)) == nn.LSTM:
            args = (module.input_size, module.hidden_size, module.num_layers, module.bias, module.batch_first, module.dropout, module.bidirectional, module.proj_size)
            xbnn_layer = xbnn.LSTM(*args)
            xbnn_layer.load_state_dict(module.state_dict())
            xbnn_layer._xb_inference = False # add attribute for xb evaluation for proper toggling later
            setattr(model, name, xbnn_layer)
        else:
            replace_all_layers_stateless(model=module, exclude=exclude)

def replace_all_layers_stateful(original_model, wage_quantize):
    """
        Recursively replace all modules within a model with XBTorch modules for stateful operation. 
        Supports wage quantization.
    """
    new_model = []
    if (wage_quantize):
        
        wage_params = get_xbtorch_param('wage_params')
        quantizer_act_error = wage_params['quantizer_act_error']
        wl_activation = wage_params['wl_activation']
        wl_error = wage_params['wl_error']
        wl_weight = wage_params['wl_weight']

        new_model.append(quantizer_act_error(wl_activation, -1))
        
    for module in original_model.model:
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
            xbnn_layer = xbnn_layer.to(next(module.parameters()).device)
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

    if (wage_quantize and new_model[-1] not in xbtorch.activation_types): 
        # add act/error quantizer if the last module is an activation
        new_model.append(quantizer_act_error(-1, wl_error))

    if (hasattr(original_model, 'model')): 
        original_model.model = nn.Sequential(*new_model)

    if (wage_quantize):
        # wage parameters
        original_model.weight_scale = {}
        original_model.weight_acc = {}
        for name, param in original_model.named_parameters():
            if ("weight" in name): wage_init.wage_init_(param, wl_weight, factor=1.0)
            param.weight_acc = param.data

def toggle_xb_eval_all_layers_stateless(model, enable):
    """
        Recursively enable or disable linear modules within a model with some customized layer module.
    """
    for _, module in model.named_children():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d) or isinstance(module, nn.RNN) or isinstance(module, nn.LSTM):
            module._xb_inference = enable # add attribute for xb evaluation for proper toggling later
        # add elifs for other supported layers
        else:
            toggle_xb_eval_all_layers_stateless(module, enable)
