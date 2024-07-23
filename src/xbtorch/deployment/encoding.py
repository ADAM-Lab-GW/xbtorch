import torch
import numpy as np 
import math

# Weight encoding functions - how should a software weight matrix be converted to conductance matrices
def encode_simple(accelerator, sw_weight, pos_idxs=[], neg_idxs=[]):

    '''
    Given a weight matrix W, return conductances matrices G_pos and G_neg such that W ∝ (G_pos - G_neg)
    '''

    Gpos = torch.clone(sw_weight)

    Gpos[Gpos > 0] = accelerator.g_max
    Gpos[Gpos < 0] = accelerator.g_min
    Gpos[Gpos == 0] = accelerator.g_min

    Gneg = torch.clone(sw_weight)
    
    Gneg[Gneg > 0] = accelerator.g_min
    Gneg[Gneg < 0] = accelerator.g_max
    Gneg[Gneg == 0] = accelerator.g_min

    # Create copies corresponding to the times the matrix would be mapped
    return [Gpos for _ in range(len(pos_idxs))], [Gneg for _ in range(len(neg_idxs))]

def encode_MAO(accelerator, sw_weight, pos_idxs=[], neg_idxs=[]):
    # Algorithm 1 - Mapping Algorithm with innter fault-tOlerance from: 
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7858421
    print('performing MAO')

    assert (len(pos_idxs) == len(neg_idxs))

    # Initialize
    Gposs = [torch.zeros_like(sw_weight) for _ in range(len(pos_idxs))]
    Gnegs = [torch.zeros_like(sw_weight) for _ in range(len(neg_idxs))]

    # Have the defect map information in an easy-to-deal-with format
    defect_map_zipped = zip(accelerator.defect_map[0][0], accelerator.defect_map[0][1])

    # Params for sensing conductances correctly
    gnorm =  (accelerator.g_max - accelerator.g_min)
    alpha = abs(torch.unique(sw_weight)[-1])

    # Step size for G adjustment
    states = 100
    G_states = np.linspace(accelerator.g_min, accelerator.g_max, states)

    # Loop over network parameters
    # Line 1-10
    for i in range(sw_weight.shape[0]):
        for j in range(sw_weight.shape[1]):
            # Loop over corresponding devices that would map this particular sw_parameter
            # first, for Gpos matrices
            for k in range(len(pos_idxs)):
                Gpos_device_idx = (i + pos_idxs[k][0], j + pos_idxs[k][1])
                if (Gpos_device_idx not in defect_map_zipped):
                    Gposs[k][i, j] = accelerator.g_max
                else: 
                    # if stuck, mirror
                    Gposs[k][i, j] = accelerator.chip[Gpos_device_idx[0], Gpos_device_idx[1]]

                # then, for Gneg matrices
                Gneg_device_idx = (i + neg_idxs[k][0], j + neg_idxs[k][1])
                if (Gneg_device_idx not in defect_map_zipped):
                    Gnegs[k][i, j] = accelerator.g_min
                else:
                    # if stuck, mirror
                    Gnegs[k][i, j] = accelerator.chip[Gneg_device_idx[0], Gneg_device_idx[1]]
    
    # Line 11-19
    for i in range(sw_weight.shape[0]):
        for j in range(sw_weight.shape[1]):
            sw_parameter = sw_weight[i, j]

            for k in range(len(pos_idxs)):
                Gpos_device_idx = (i + pos_idxs[k][0], j + pos_idxs[k][1])
                Gpos_stuck = Gpos_device_idx in defect_map_zipped

                Gneg_device_idx = (i + neg_idxs[k][0], j + neg_idxs[k][1])
                Gneg_stuck = Gneg_device_idx in defect_map_zipped

                if (Gpos_stuck and Gneg_stuck):
                    continue

                # adjust Gpos[k][i, j], then adjust Gneg[k][i, j]
                if (not Gpos_stuck):
                    # Minimize diff (Line 16)
                    diff = math.inf
                    for adjusted_G in G_states:
                        # update G temporarily
                        Gposs[k][i, j] = adjusted_G

                        # Compute diff, see if there is an improvement
                        Gpos_sum = 0
                        Gneg_sum = 0
                        for l in range(len(pos_idxs)): Gpos_sum += Gposs[l][i, j]
                        for l in range(len(neg_idxs)): Gneg_sum += Gnegs[l][i, j]
                        newDiff = abs(sw_parameter - ((Gpos_sum - Gneg_sum) / gnorm) * alpha)

                        if (newDiff < diff):
                            diff = newDiff
                            optimal_G = adjusted_G

                    Gposs[k][i, j] = optimal_G

                if (not Gneg_stuck):
                    diff = math.inf
                    for adjusted_G in G_states:
                        # update G temporarily
                        Gnegs[k][i, j] = adjusted_G

                        # Compute diff, see if there is an improvement
                        Gpos_sum = 0
                        Gneg_sum = 0
                        for l in range(len(pos_idxs)): Gpos_sum += Gposs[l][i, j]
                        for l in range(len(neg_idxs)): Gneg_sum += Gnegs[l][i, j]
                        newDiff = abs(sw_parameter - ((Gpos_sum - Gneg_sum) / gnorm) * alpha)

                        if (newDiff < diff):
                            diff = newDiff
                            optimal_G = adjusted_G

                    Gnegs[k][i, j] = optimal_G

    return Gposs, Gnegs



    multibit_debug = False
    multibit_debug_clip = False
    
    defect_map_zipped = zip(accelerator.defect_map[0][0], accelerator.defect_map[0][1])
    for i in range(sw_weight.shape[0]):
        for j in range(sw_weight.shape[1]):
            sw_parameter = sw_weight[i, j]
            for k in range(len(pos_idxs)): # for each (Gpos, Gneg) conductance matrix pair
                # what's the array index?
                # first, for Gpos
                Gpos_idx = (i + pos_idxs[k][0], j + pos_idxs[k][1])
                Gneg_idx = (i + neg_idxs[k][0], j + neg_idxs[k][1])

                # get stuck status
                Gpos_stuck = Gpos_idx in defect_map_zipped
                Gneg_stuck = Gneg_idx in defect_map_zipped

                if (Gpos_stuck and Gneg_stuck):# or (not Gpos_stuck and not Gneg_stuck)):
                    # if both are stuck, or both are not stuck, skip the MAO operation since diff is already as good as it can be
                    continue

                gnorm =  (accelerator.g_max - accelerator.g_min)
                alpha = abs(torch.unique(sw_weight)[-1])

                diff = abs(sw_parameter - ((accelerator.chip[Gpos_idx[0], Gpos_idx[1]] - accelerator.chip[Gneg_idx[0], Gneg_idx[1]]) / gnorm) * alpha)

                # Does one get stuck? If yes, adjust to minimize diff
                if (not Gpos_stuck):
                    # adjust the device
                    # what if we allowed more than 2 states
                    if (multibit_debug): 
                        accelerator.chip[Gpos_idx[0], Gpos_idx[1]] = (sw_parameter / alpha ) * gnorm + accelerator.chip[Gneg_idx[0], Gneg_idx[1]]
                        if (multibit_debug_clip): accelerator.chip[Gpos_idx[0], Gpos_idx[1]] = np.clip(accelerator.chip[Gpos_idx[0], Gpos_idx[1]], accelerator.g_min, accelerator.g_max)
                    else:
                        if abs(sw_parameter - ((accelerator.g_max - accelerator.chip[Gneg_idx[0], Gneg_idx[1]]) / gnorm) * alpha) < diff:
                            accelerator.chip[Gpos_idx[0], Gpos_idx[1]] = accelerator.g_max

                        if abs(sw_parameter - ((accelerator.g_min - accelerator.chip[Gneg_idx[0], Gneg_idx[1]]) / gnorm) * alpha) < diff:
                            accelerator.chip[Gpos_idx[0], Gpos_idx[1]] = accelerator.g_min


                elif (not Gneg_stuck):

                    if (multibit_debug):
                        accelerator.chip[Gneg_idx[0], Gneg_idx[1]] =  - ((sw_parameter / alpha ) * gnorm - accelerator.chip[Gpos_idx[0], Gpos_idx[1]])
                        if (multibit_debug_clip): accelerator.chip[Gneg_idx[0], Gneg_idx[1]] = np.clip(accelerator.chip[Gneg_idx[0], Gneg_idx[1]], accelerator.g_min, accelerator.g_max)
                    else:
                        if abs(sw_parameter - ((accelerator.chip[Gpos_idx[0], Gpos_idx[1]] - accelerator.g_max) / gnorm) * alpha) < diff:
                            accelerator.chip[Gneg_idx[0], Gneg_idx[1]] = accelerator.g_max
                        if abs(sw_parameter - ((accelerator.chip[Gpos_idx[0], Gpos_idx[1]] - accelerator.g_min)  / gnorm) * alpha) < diff:
                            accelerator.chip[Gneg_idx[0], Gneg_idx[1]] = accelerator.g_min

    # Add back defect map information
    accelerator.chip[accelerator.defect_map[0]] = accelerator.defect_map[1]