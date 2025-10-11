"""
Decorators for patching PyTorch models to make them compatible
with XBTorch operations.
"""
from xbtorch import get_xbtorch_param

from .utils import replace_all_layers_stateful, replace_all_layers_stateless, toggle_xb_eval_all_layers_stateless

def xbtorch_model(original_model, replace_all=False, exclude=None):
    """
    Patch a PyTorch model for XBTorch compatibility.

    This function takes an existing model (with a ``.model`` attribute that is
    an ``nn.Sequential`` or similar container) and replaces supported layers
    (e.g., ``nn.Linear``, ``nn.Conv2d``, ``nn.RNN``, ``nn.LSTM``) with their
    XBTorch equivalents. It preserves trained parameters by copying the
    state dictionary, and it attaches additional functionality for 
    hardware-aware training and inference.

    Features
    --------
    - **Device simulation**: Layers are converted into XBTorch versions that
      support device-aware weight updates (noise, variability).
    - **WAGE quantization**: If enabled during initialization, activation,
      error, and weight quantization modules are automatically inserted.
    - **Inference accelerator integration**: If an inference accelerator was
      initialized, the returned model gains methods to toggle hardware-aware
      inference and to map weights onto simulated crossbar arrays.
    - **Seamless PyTorch API**: All layers remain compatible with PyTorch
      training, optimizers, and loss functions.

    Parameters
    ----------
    original_model : torch.nn.Module
        A PyTorch model instance with a ``.model`` attribute (typically an
        ``nn.Sequential``) that contains the layers to be patched.

    Returns
    -------
    model : torch.nn.Module
        The patched model. Additional methods may be attached depending on
        initialization parameters:
        
        - ``model.xb_eval(enable=True/False)``  
          Toggle hardware-aware inference mode.
        - ``model.initialize_array_mappings(output_polling_mode='avg', ...)``  
          Map weights onto the simulated accelerator crossbar.
        - ``model.get_array_mappings()``  
          Retrieve conductance array mappings for inspection or reuse.

    Raises
    ------
    RuntimeError
        If XBTorch has not been initialized with :func:`xbtorch.initialize`.
    RuntimeError
        If the provided model does not have a ``.model`` attribute.
    ValueError
        If an encountered PyTorch layer type is not supported by XBTorch.

    Notes
    -----
    - Unsupported modules are kept as-is, but a warning is printed.
    - Quantization requires WAGE parameters (bit-widths, rounding) to have
      been set during initialization.
    - Inference accelerator mappings assume that crossbar dimensions and
      encoding/mapping schemes are defined during initialization.
    """
    # Copies state dictionary as well
    if (not get_xbtorch_param('initialized')): raise RuntimeError('XBTorch needs to be initialized, please refer to API for instructions.')

    wage_quantize = get_xbtorch_param('wage_quantize')
    xb_inference_accelerator = get_xbtorch_param('inference_accelerator')
    original_model.xb_forward = False # declare this to be false, xb_eval() has to be explicitly called
    
    # How to replace layers? stateless or stateful.
    if (replace_all):
        # stateless
        replace_all_layers_stateless(model=original_model,
                                     exclude=[] if exclude is None else exclude)
    else:
        # stateful
        replace_all_layers_stateful(model=original_model, wage_quantize=wage_quantize)

    if (xb_inference_accelerator):
        # if initialization included an inference accelerator
        def toggle(enable=True):
            original_model.xb_forward = enable
            if not replace_all:
                # stateless
                for module in original_model.model:
                    module._xb_inference = enable
            else:
                toggle_xb_eval_all_layers_stateless(original_model, enable=enable)

        def initialize_array_mappings(output_polling_mode='avg', existing_mappings=[], additional_args={}):
            
            if replace_all:
                print("Can not map to array in stateless operation.")
                return
            
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