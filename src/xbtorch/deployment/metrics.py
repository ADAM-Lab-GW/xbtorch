import torch
import numpy as np 
import math
import xbtorch

def compute_error(model):
    errors = []
    for module in model.model:
        if (hasattr(module, '_array_mappings')):
            error = error_mapping(module.inference_accelerator, module.weight.data, 
                                  module._array_mappings['Gpos'], 
                                  module._array_mappings['Gneg'], 
                                  module._array_mappings['output_polling_mode'])
            errors.append(error)

    return errors

# Weight encoding functions - how should a software weight matrix be converted to conductance matrices
def error_mapping(accelerator, sw_weight, pos_idxs=[], neg_idxs=[], polling_mode='avg'):

    alpha = torch.unique(sw_weight)[-1].item()

    cideal = np.array(sw_weight)
    gposs = [] 
    gnegs = []
    gnorm = accelerator.g_max - accelerator.g_min

    for pos_idx in pos_idxs:
        gpos = accelerator.chip[pos_idx[0]:pos_idx[0]+sw_weight.shape[0], pos_idx[1]:pos_idx[1]+sw_weight.shape[1]]
        gposs.append(gpos)

    for neg_idx in neg_idxs:
        gneg = accelerator.chip[neg_idx[0]:neg_idx[0]+sw_weight.shape[0], neg_idx[1]:neg_idx[1]+sw_weight.shape[1]]
        gnegs.append(gneg)

    if (polling_mode == 'sum'):
        cmapped = (np.sum(gposs, axis=0) - np.sum(gnegs, axis=0)) / gnorm * alpha
    else:
        cmapped = (np.average(gposs, axis=0) - np.average(gnegs, axis=0) ) / gnorm * alpha

    cmapped = cmapped.flatten()
    cideal = cideal.flatten()

    error = np.linalg.norm(cmapped - cideal) / np.linalg.norm(cideal)

    return error * 100