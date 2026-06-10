"""
Weight encoding schemes for mapping software weight matrices
to crossbar conductance arrays.

This module implements different strategies for converting
software weights into positive/negative conductance matrices
(`G_pos`, `G_neg`), including support for fault tolerance in the
presence of device-level stuck-at defects.

Functions
---------
- :func:`encode_simple_binary` : Basic 2-state binary encoding.
- :func:`encode_LEA1` : Layer Ensemble Averaging (LEA-1) with row masking.
- :func:`encode_LEA2` : Layer Ensemble Averaging (LEA-2) with defect-aware
  adjustments.
- :func:`encode_MAO` : Mapping Algorithm with inner Fault-tOlerance (MAO).
- :func:`quantize_to_nearest` : Utility for nearest-level conductance quantization.

References
----------
- LEA schemes: D. Niu et al., *"Ensemble Learning for Memristive Neural
  Networks"*, ACM, 2023.
- MAO scheme: C. Yu et al., *"Mapping Algorithm With Fault-Tolerance for
  Memristor Crossbar"*, IEEE TVLSI, 2016.
"""

import torch
import numpy as np 
import math
from .metrics import error_mapping

# Weight encoding functions - how should a software weight matrix be converted to conductance matrices
def encode_simple_binary(accelerator, sw_weight, pos_idxs=[], neg_idxs=[], additional_args={}):
    """
    Encode weights using a simple binary scheme.

    Each software weight is represented by two possible conductance states:
    - Positive weights → `G_pos = g_max`, `G_neg = g_min`
    - Negative weights → `G_pos = g_min`, `G_neg = g_max`
    - Near-zero weights → `G_pos = G_neg = g_min`

    Parameters
    ----------
    accelerator : GenericAccelerator
        Crossbar accelerator instance providing hardware parameters
        (`g_min`, `g_max`).
    sw_weight : torch.Tensor
        Software weight matrix.
    pos_idxs, neg_idxs : list of tuple[int, int], optional
        Array mapping positions for positive and negative conductances.
    additional_args : dict, optional
        Extra parameters, supports:
        - ``zero_tol`` (float): Threshold below which weights are treated as zero.

    Returns
    -------
    list[torch.Tensor], list[torch.Tensor]
        Lists of positive and negative conductance matrices.
    """
    zero_tol = additional_args.get("zero_tol", 1e-3) # 0.001

    Gpos = torch.clone(sw_weight)

    Gpos[Gpos > zero_tol] = accelerator.g_max                    # significantly positive
    Gpos[Gpos < -zero_tol] = accelerator.g_min                   # significantly negative
    Gpos[torch.abs(Gpos) < zero_tol] = accelerator.g_min          # close to zero

    Gneg = torch.clone(sw_weight)
    Gneg[torch.abs(Gneg) < zero_tol] = accelerator.g_min          # close to zero
    Gneg[Gneg > zero_tol] = accelerator.g_min                    # significantly positive
    Gneg[Gneg < -zero_tol] = accelerator.g_max                   # significantly negative

    # Create copies corresponding to the times the matrix would be mapped
    return [torch.clone(Gpos) for _ in range(max(1, len(pos_idxs)))], [torch.clone(Gneg) for _ in range(max(1, len(pos_idxs)))]

# Weight encoding functions - how should a software weight matrix be converted to conductance matrices
def encode_LEA1(accelerator, sw_weight, pos_idxs=[], neg_idxs=[], additional_args={}):
    """
    Encode weights using Layer Ensemble Averaging (LEA-1).

    In LEA-1, each weight is redundantly mapped across multiple
    crossbar rows/columns (beta copies). Rows corresponding to the most
    defective devices are masked out, preserving accuracy while reducing
    hardware variability impact.

    Parameters
    ----------
    accelerator : GenericAccelerator
        Crossbar accelerator instance.
    sw_weight : torch.Tensor
        Software weight matrix.
    pos_idxs, neg_idxs : list of tuple[int, int]
        Mapping indices for redundant weight placement.
    additional_args : dict
        Must include:
        - ``alpha`` (int): Number of rows to keep.
        - ``beta`` (int): Total number of redundant rows.

    Returns
    -------
    list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor
        Positive and negative conductance matrices, and corresponding row masks.
    """
    # Necessary
    beta = additional_args['beta']
    alpha = additional_args['alpha']

    assert alpha >= 1 and alpha <= beta

    discard_row_count =  beta - alpha

    Gposs, Gnegs = encode_simple_binary(accelerator, sw_weight, pos_idxs=[0], neg_idxs=[0]) # indices are dummy arrays of len 1
    ideal_Gs = [Gposs[0], Gnegs[0]]

    # let's also retrieve the \beta defective copies
    defective_Gss = [*encode_simple_binary(accelerator, sw_weight, pos_idxs, neg_idxs)] # no longer a dummy array; actual mapped idxs, but these matrices are still ideal since encode_simple_binary doesn't account for chip defects

    # Have the defect map information in an easy-to-deal-with format
    defect_map_zipped = list(zip(accelerator.defect_map[0][0], accelerator.defect_map[0][1]))
    # In-fact, let's make it a hash table to make the lookups be constant, leads to a high improvement in program run-time
    defect_map_zipped = set(defect_map_zipped)

    for i in range(sw_weight.shape[0]):
        for j in range(sw_weight.shape[1]):
            # Loop over corresponding devices that would map this particular sw_parameter
            # first, for Gpos matrices
            for k in range(beta):
                Gpos_device_idx = (i + pos_idxs[k][0], j + pos_idxs[k][1])

                if (Gpos_device_idx in defect_map_zipped):
                    defective_Gss[0][k][i, j] = accelerator.read_chip(Gpos_device_idx[0], 1, Gpos_device_idx[1], 1)

                Gneg_device_idx = (i + neg_idxs[k][0], j + neg_idxs[k][1])
                if (Gneg_device_idx in defect_map_zipped):
                    defective_Gss[1][k][i, j] = accelerator.read_chip(Gneg_device_idx[0], 1, Gneg_device_idx[1], 1)

    # next, we make these defective copies mirror actual defects from our simulated chip. In a HW implementation, this is a simple read operation.
    # # Loop over network parameters
    # Line 1-10: Unvectorized - much slower
    masks = []
    for idx, G in enumerate(ideal_Gs):
        # A = ideal_Gs[idx]
        defective_Gs = defective_Gss[idx]

        # calculated summed G variation
        diffs = []

        for i in range(G.shape[0]):
            row_diffs = [torch.sum(torch.abs(G[i] - B[i])) for B in defective_Gs]
            diffs.append(torch.Tensor(row_diffs))

        diffs = torch.stack(diffs)

        _, rankings = torch.sort(diffs, dim=1)  # Rank in ascending order (lowest difference first)
        mask = torch.ones((beta, G.shape[0]), device=sw_weight.device)

        if (discard_row_count > 0):
            for i in range(G.shape[0]):
                # Identify the index of the most defective matrix for the current row
                worst_index = rankings[i, -discard_row_count:]  # Last alpha indices corresponds to the alpha most defective
                # Set the mask value for the alpha defective matrix to 0
                mask[worst_index, i] = 0

        # Masks computed
        # TODO: Filenames for LEA should have alpha incorporated
        masks.append(mask)

    # Finally, return the original Gposs/Gnegs (still encode_simple_binary), and the determined masks
    Gposs, Gnegs = encode_simple_binary(accelerator, sw_weight, pos_idxs, neg_idxs)
    return Gposs, Gnegs, masks[0], masks[1]


def encode_LEA2(accelerator, sw_weight, pos_idxs=[], neg_idxs=[], states=2**1, additional_args={}, log=False):
    """
    Encode weights using Layer Ensemble Averaging (LEA-2).

    LEA-2 adaptively adjusts conductances in the presence of
    stuck-at defects. Each weight is distributed across redundant
    devices, with defective elements compensated by tuning remaining
    devices toward nearest available conductance states.

    Parameters
    ----------
    accelerator : GenericAccelerator
        Crossbar accelerator instance.
    sw_weight : torch.Tensor
        Software weight matrix.
    pos_idxs, neg_idxs : list of tuple[int, int]
        Mapping indices for redundant weight placement.
    states : int, optional
        Number of available conductance states (default: 2).
    additional_args : dict, optional
        Extra configuration parameters.
    log : bool, optional
        If True, prints debugging information during adjustment.

    Returns
    -------
    list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor
        Positive and negative conductance matrices, and row masks.
    """
    assert (len(pos_idxs) == len(neg_idxs))

    # Initialize
    Gposs = np.zeros((len(pos_idxs), *sw_weight.shape))
    Gnegs = np.zeros((len(neg_idxs), *sw_weight.shape))

    # Have the defect map information in an easy-to-deal-with format
    defect_map_zipped = list(zip(accelerator.defect_map[0][0], accelerator.defect_map[0][1]))
    # In-fact, let's make it a hash table to make the lookups be constant, leads to a high improvement in program run-time
    defect_map_zipped = set(defect_map_zipped)

    # Params for sensing conductances correctly
    gnorm =  (accelerator.g_max - accelerator.g_min)
    alpha = abs(torch.unique(sw_weight)[-1])

    G_states = np.linspace(accelerator.g_min, accelerator.g_max, states)
    
    # # Loop over network parameters
    # Line 1-10: Unvectorized - much slower
    for i in range(sw_weight.shape[0]):
        for j in range(sw_weight.shape[1]):
            # Loop over corresponding devices that would map this particular sw_parameter
            # first, for Gpos matrices
            for k in range(len(pos_idxs)):
                Gpos_device_idx = (i + pos_idxs[k][0], j + pos_idxs[k][1])

                if (Gpos_device_idx not in defect_map_zipped):
                    Gposs[k, i, j] = accelerator.g_max
                else:
                    # if stuck, mirror
                    Gposs[k, i, j] = accelerator.read_chip(Gpos_device_idx[0], 1, Gpos_device_idx[1], 1)

                # then, for Gneg matrices
                Gneg_device_idx = (i + neg_idxs[k][0], j + neg_idxs[k][1])
                if (Gneg_device_idx not in defect_map_zipped):
                    Gnegs[k, i, j] = accelerator.g_min
                else:
                    # if stuck, mirror
                    Gnegs[k, i, j] = accelerator.read_chip(Gneg_device_idx[0], 1, Gneg_device_idx[1], 1)
    
    beta = len(pos_idxs)

    # Line 11-19
    for i in range(sw_weight.shape[0]):
        for j in range(sw_weight.shape[1]):
            sw_parameter = sw_weight[i, j]

            if (log):
                print('\n', i, j)
                print('original sw parameter', sw_parameter)
                print('original HW parameter', (np.average(Gposs[:, i, j]) - np.average(Gnegs[:, i, j])) / gnorm * alpha)
                print('original Gs', Gposs[:, i, j], Gnegs[:, i, j])

            for k in range(len(pos_idxs)):
                Gpos_device_idx = (i + pos_idxs[k][0], j + pos_idxs[k][1])
                Gpos_stuck = Gpos_device_idx in defect_map_zipped

                Gneg_device_idx = (i + neg_idxs[k][0], j + neg_idxs[k][1])
                Gneg_stuck = Gneg_device_idx in defect_map_zipped

                if (Gpos_stuck and Gneg_stuck):
                    if (log): print('both stuck, continue')
                    continue

                if not Gpos_stuck:
                    if log: print(k, 'adjusting Gpos', Gposs[k, i, j])


                    diffs = []
                    for candidate_G in G_states:
                        diff = np.abs(sw_parameter - (( (np.average(Gposs[:, i, j]) - Gposs[k, i, j]/beta + candidate_G/beta) - np.average(Gnegs[:, i, j])) / gnorm * alpha))
                        diffs.append(diff)

                    optimal_G = G_states[np.argmin(diffs)]
                    optimal_G = quantize_to_nearest(x=optimal_G, G_min=G_states[0], d=G_states[1]-G_states[0], n=states)
                    Gposs[k, i, j] = optimal_G
                    if log: print(k, 'adjusted Gpos', Gposs[k, i, j])

                if not Gneg_stuck:
                    if log: print(k, 'adjusting Gneg', Gnegs[k, i, j])

                    diffs = []
                    for candidate_G in G_states:
                        diff = np.abs(sw_parameter - ((np.average(Gposs[:, i, j]) - (np.average(Gnegs[:, i, j]) - Gnegs[k, i, j]/beta + candidate_G/beta)) / gnorm * alpha))
                        diffs.append(diff)

                    optimal_G = G_states[np.argmin(diffs)]
                    optimal_G = quantize_to_nearest(x=optimal_G, G_min=G_states[0], d=G_states[1]-G_states[0], n=states)
                    Gnegs[k, i, j] = optimal_G
                    if log: print(k, 'adjusted Gneg', Gnegs[k, i, j])

            if (log):
                print('adjusted sw parameter', sw_parameter)
                print('adjusted HW parameter', (np.average(Gposs[:, i, j]) - np.average(Gnegs[:, i, j])) / gnorm * alpha)
                exit()

    # convert to pytorch tensors
    Gposs = [torch.from_numpy(Gposs[k]) for k in range(Gposs.shape[0])]
    Gnegs = [torch.from_numpy(Gnegs[k]) for k in range(Gnegs.shape[0])]

    # return Gposs, Gnegs

    masks = []
    mask = torch.ones((len(pos_idxs), sw_weight.shape[0]))
    # Masks computed
    # Same for Gpos and Gneg in this scheme
    masks.append(mask)
    masks.append(torch.clone(mask))

    # Finally, return the original Gposs/Gnegs (still encode_simple_binary), and the determined masks
    return Gposs, Gnegs, masks[0], masks[1]

def quantize_to_nearest(x, G_min, d, n):
    """
    Quantize a conductance value to the nearest available state.

    Parameters
    ----------
    x : float
        Target conductance value.
    G_min : float
        Minimum conductance.
    d : float
        Step size between quantization levels.
    n : int
        Number of quantization levels.

    Returns
    -------
    float
        Quantized conductance value.
    """
    return G_min + np.clip(round((x - G_min) / d), 0, n-1) * d

def encode_MAO(accelerator, sw_weight, pos_idxs=[], neg_idxs=[], states=2**1, additional_args={}, log=False):
    """
    Encode weights using the Mapping Algorithm with inner Fault-tOlerance (MAO).

    MAO (Yu et al., 2016) is a defect-aware mapping scheme
    where conductance values are analytically adjusted in redundant
    devices to approximate the target software weight even when some
    devices are stuck.

    Parameters
    ----------
    accelerator : GenericAccelerator
        Crossbar accelerator instance.
    sw_weight : torch.Tensor
        Software weight matrix.
    pos_idxs, neg_idxs : list of tuple[int, int]
        Mapping indices for redundant weight placement.
    states : int, optional
        Number of available conductance states (default: 2).
    additional_args : dict, optional
        Extra configuration parameters.
    log : bool, optional
        If True, prints debugging information during adjustment.

    Returns
    -------
    list[torch.Tensor], list[torch.Tensor]
        Adjusted positive and negative conductance matrices.
    """
    assert (len(pos_idxs) == len(neg_idxs))

    # Initialize
    Gposs = np.zeros((len(pos_idxs), *sw_weight.shape))
    Gnegs = np.zeros((len(neg_idxs), *sw_weight.shape))

    # Have the defect map information in an easy-to-deal-with format
    defect_map_zipped = list(zip(accelerator.defect_map[0][0], accelerator.defect_map[0][1]))
    # In-fact, let's make it a hash table to make the lookups be constant, leads to a high improvement in program run-time
    defect_map_zipped = set(defect_map_zipped)

    # Params for sensing conductances correctly
    gnorm =  (accelerator.g_max - accelerator.g_min)
    alpha = abs(torch.unique(sw_weight)[-1])

    G_states = np.linspace(accelerator.g_min, accelerator.g_max, states)
    
    # # Loop over network parameters
    # Line 1-10: Unvectorized - much slower
    for i in range(sw_weight.shape[0]):
        for j in range(sw_weight.shape[1]):
            # Loop over corresponding devices that would map this particular sw_parameter
            # first, for Gpos matrices
            for k in range(len(pos_idxs)):
                Gpos_device_idx = (i + pos_idxs[k][0], j + pos_idxs[k][1])

                if (Gpos_device_idx not in defect_map_zipped):
                    Gposs[k, i, j] = accelerator.g_max
                else:
                    # if stuck, mirror
                    Gposs[k, i, j] = accelerator.read_chip(Gpos_device_idx[0], 1, Gpos_device_idx[1], 1)

                # then, for Gneg matrices
                Gneg_device_idx = (i + neg_idxs[k][0], j + neg_idxs[k][1])
                if (Gneg_device_idx not in defect_map_zipped):
                    Gnegs[k, i, j] = accelerator.g_min
                else:
                    # if stuck, mirror
                    Gnegs[k, i, j] = accelerator.read_chip(Gneg_device_idx[0], 1, Gneg_device_idx[1], 1)
    
    # Line 11-19
    for i in range(sw_weight.shape[0]):
        for j in range(sw_weight.shape[1]):
            sw_parameter = sw_weight[i, j]

            if (log):
                print('\n', i, j)
                print('original sw parameter', sw_parameter)
                print('original HW parameter', (np.sum(Gposs[:, i, j]) - np.sum(Gnegs[:, i, j])) / gnorm * alpha)

            for k in range(len(pos_idxs)):
                Gpos_device_idx = (i + pos_idxs[k][0], j + pos_idxs[k][1])
                Gpos_stuck = Gpos_device_idx in defect_map_zipped

                Gneg_device_idx = (i + neg_idxs[k][0], j + neg_idxs[k][1])
                Gneg_stuck = Gneg_device_idx in defect_map_zipped

                if (Gpos_stuck and Gneg_stuck):
                    if (log): print('both stuck, continue')
                    continue

                # analytically perform the adjustment
                if not Gpos_stuck:
                    if log: print(k, 'adjusting Gpos', Gposs[k, i, j])
                    Gpos_sum = np.sum(Gposs[:, i, j])
                    Gneg_sum = np.sum(Gnegs[:, i, j])

                    optimal_G = (sw_parameter / alpha) * gnorm + Gneg_sum - Gpos_sum + Gposs[k, i, j]
                    optimal_G = quantize_to_nearest(x=optimal_G.cpu().numpy(), G_min=G_states[0], d=G_states[1]-G_states[0], n=states)
                    Gposs[k, i, j] = optimal_G
                    if log: print(k, 'adjusted Gpos', Gposs[k, i, j])

                if not Gneg_stuck:
                    if log: print(k, 'adjusting Gneg', Gnegs[k, i, j])

                    Gpos_sum = np.sum(Gposs[:, i, j])
                    Gneg_sum = np.sum(Gnegs[:, i, j])

                    optimal_G = - ((sw_parameter / alpha) * gnorm - Gpos_sum + Gneg_sum - Gnegs[k, i, j])
                    optimal_G = quantize_to_nearest(x=optimal_G.cpu().numpy(), G_min=G_states[0], d=G_states[1]-G_states[0], n=states)
                    Gnegs[k, i, j] = optimal_G
                    if log: print(k, 'adjusted Gneg', Gnegs[k, i, j])

            if (log):
                print('adjusted sw parameter', sw_parameter)
                print('adjusted HW parameter', (np.sum(Gposs[:, i, j]) - np.sum(Gnegs[:, i, j])) / gnorm * alpha)

    # convert to pytorch tensors
    Gposs = [torch.from_numpy(Gposs[k]) for k in range(Gposs.shape[0])]
    Gnegs = [torch.from_numpy(Gnegs[k]) for k in range(Gnegs.shape[0])]

    return Gposs, Gnegs