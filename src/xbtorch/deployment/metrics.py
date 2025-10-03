"""
Error computation and weight encoding utilities for crossbar simulation.

This module provides functions to evaluate how well software weight matrices
are represented on simulated crossbar arrays. It includes tools to compute
mapping errors between software weights and their corresponding hardware
conductance matrices, as well as utilities for reading and processing
encoded weight matrices.

Functions
---------
- :func:`compute_error` : Compute hardware mapping errors for all mapped layers in a model.
- :func:`error_mapping` : Compute the error between a software weight matrix and its crossbar representation.

Notes
-----
Accurate error computation is essential for validating hardware-aware
training and mapping strategies. The module assumes that layers have
predefined array mappings and that the accelerator supports reading
conductance values.
"""

import torch
import numpy as np 
import math
import xbtorch

def compute_error(model):
    """
    Compute mapping errors for all layers of a model that have crossbar mappings.

    Iterates over the layers of the model and computes the percentage error
    between the software weight matrices and their corresponding mapped
    conductance matrices on the hardware accelerator.

    Parameters
    ----------
    model : object
        The model object containing layers with potential crossbar mappings.
        Each layer must have a `_array_mappings` attribute with keys:
        `'Gpos'`, `'Gneg'`, and `'output_polling_mode'`.

    Returns
    -------
    tuple[list[float], list[list[np.ndarray]]]
        - List of errors (percentage) for each mapped layer.
        - List of pairs `[cmapped, cideal]` for each layer, where `cmapped`
          is the crossbar-mapped matrix and `cideal` is the original software weight matrix.
    """
    errors = []
    matrices = []
    for module in model.model:
        if (hasattr(module, '_array_mappings')):
            error, matrix = error_mapping(module.inference_accelerator, 
                                  module.weight.data, 
                                  pos_idxs=module._array_mappings['Gpos'], 
                                  neg_idxs=module._array_mappings['Gneg'], 
                                  polling_mode=module._array_mappings['output_polling_mode'])
            errors.append(error)
            matrices.append(matrix)

    return errors, matrices

def error_mapping(accelerator, sw_weight, pos_idxs=[], neg_idxs=[], pos_matrices=None, neg_matrices=None, polling_mode='avg'):
    """
    Compute the error between a software weight matrix and its crossbar representation.

    Converts a software weight matrix into one or more conductance matrices
    using positive and negative subarrays, then calculates the normalized
    error compared to the ideal software weight.

    Parameters
    ----------
    accelerator : GenericAccelerator
        The accelerator instance containing chip dimensions and read/write methods.
    sw_weight : torch.Tensor
        The software weight matrix to encode.
    pos_idxs : list[tuple[int, int]], optional
        List of starting indices for positive conductance subarrays (default: []).
    neg_idxs : list[tuple[int, int]], optional
        List of starting indices for negative conductance subarrays (default: []).
    pos_matrices : list[np.ndarray], optional
        Precomputed positive conductance matrices (default: None).
    neg_matrices : list[np.ndarray], optional
        Precomputed negative conductance matrices (default: None).
    polling_mode : str, optional
        Method to combine subarray outputs: `'avg'` for averaging, `'sum'` for summation (default: 'avg').

    Returns
    -------
    tuple[list[float], list[list[np.ndarray]]]
        - List of percentage errors for each conductance matrix mapping.
        - List of `[cmapped, cideal]` pairs for each mapping.

    Notes
    -----
    - `gnorm` is computed as `g_max - g_min` of the accelerator.
    - The final mapped matrix is normalized and compared to the ideal software weight.
    - Supports both precomputed matrices and reading directly from the accelerator.
    """
    gnorm_scales = [1.0]
    # gnorm_scales = np.logspace(-1, 1, 25)
    errors = []
    matrices = []

    alpha = torch.unique(sw_weight)[-1].item()

    cideal = np.array(sw_weight)

    gnorm = accelerator.g_max - accelerator.g_min

    if (pos_matrices == None and neg_matrices == None):

        gposs = [] 
        gnegs = []

        for pos_idx in pos_idxs:
            # gpos = accelerator._chip[pos_idx[0]:pos_idx[0]+sw_weight.shape[0], pos_idx[1]:pos_idx[1]+sw_weight.shape[1]]
            gpos = accelerator.read_chip(pos_idx[0], sw_weight.shape[0], pos_idx[1], sw_weight.shape[1])
            gposs.append(gpos)

        for neg_idx in neg_idxs:
            # gneg = accelerator.chip[neg_idx[0]:neg_idx[0]+sw_weight.shape[0], neg_idx[1]:neg_idx[1]+sw_weight.shape[1]]
            gneg = accelerator.read_chip(neg_idx[0], sw_weight.shape[0], neg_idx[1], sw_weight.shape[1])
            gnegs.append(gneg)
    else:
        gposs = pos_matrices
        gnegs = neg_matrices

    for gnorm_scale in gnorm_scales:

        if (polling_mode == 'sum'):
            cmapped = (np.sum(gposs, axis=0) - np.sum(gnegs, axis=0)) / (gnorm * gnorm_scale) * alpha
        else:
            cmapped = (np.average(gposs, axis=0) - np.average(gnegs, axis=0) ) / (gnorm * gnorm_scale) * alpha

        matrices.append([cmapped, cideal])

        cmapped = cmapped.flatten()
        cideal = cideal.flatten()

        error = np.linalg.norm(cmapped - cideal) / np.linalg.norm(cideal)
        errors.append(error * 100)


    return errors, matrices