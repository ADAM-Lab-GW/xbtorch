import torch
import numpy as np 
import math
import xbtorch

def compute_error(model):
    errors = []
    for module in model.model:
        if (hasattr(module, '_array_mappings')):
            error = error_mapping(module.inference_accelerator, 
                                  module.weight.data, 
                                  pos_idxs=module._array_mappings['Gpos'], 
                                  neg_idxs=module._array_mappings['Gneg'], 
                                  polling_mode=module._array_mappings['output_polling_mode'])
            errors.append(error)

    return errors

# Weight encoding functions - how should a software weight matrix be converted to conductance matrices
def error_mapping(accelerator, sw_weight, pos_idxs=[], neg_idxs=[], pos_matrices=None, neg_matrices=None, polling_mode='avg'):

    gnorm_scales = [1.0]
    # gnorm_scales = np.logspace(-1, 1, 25)
    errors = []

    alpha = torch.unique(sw_weight)[-1].item()

    cideal = np.array(sw_weight)

    gnorm = accelerator.g_max - accelerator.g_min

    if (pos_matrices == None and neg_matrices == None):

        gposs = [] 
        gnegs = []

        for pos_idx in pos_idxs:
            gpos = accelerator.chip[pos_idx[0]:pos_idx[0]+sw_weight.shape[0], pos_idx[1]:pos_idx[1]+sw_weight.shape[1]]
            gposs.append(gpos)

        for neg_idx in neg_idxs:
            gneg = accelerator.chip[neg_idx[0]:neg_idx[0]+sw_weight.shape[0], neg_idx[1]:neg_idx[1]+sw_weight.shape[1]]
            gnegs.append(gneg)
    else:
        gposs = pos_matrices
        gnegs = neg_matrices

    for gnorm_scale in gnorm_scales:

        if (polling_mode == 'sum'):
            cmapped = (np.sum(gposs, axis=0) - np.sum(gnegs, axis=0)) / (gnorm * gnorm_scale) * alpha
        else:
            cmapped = (np.average(gposs, axis=0) - np.average(gnegs, axis=0) ) / (gnorm * gnorm_scale) * alpha

        cmapped = cmapped.flatten()
        cideal = cideal.flatten()

        error = np.linalg.norm(cmapped - cideal) / np.linalg.norm(cideal)
        errors.append(error * 100)

    return errors