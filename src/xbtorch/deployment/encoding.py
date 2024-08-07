import torch
import numpy as np 
import math
from .metrics import error_mapping

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

def quantize_to_nearest(x, G_min, d, n):
    # O(1)
    return G_min + np.clip(round((x - G_min) / d), 0, n-1) * d

def encode_MAO(accelerator, sw_weight, pos_idxs=[], neg_idxs=[], states=2**1, log=False):
    # Algorithm 1 - Mapping Algorithm with innter fault-tOlerance from: 
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7858421

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
                    Gposs[k, i, j] = accelerator.read_chip(Gpos_device_idx[0], 1, Gpos_device_idx[1], 1)#accelerator.chip[Gpos_device_idx[0], Gpos_device_idx[1]]
                    # Gposs[k, i, j] = accelerator._chip[Gpos_device_idx[0], Gpos_device_idx[1]]

                # then, for Gneg matrices
                Gneg_device_idx = (i + neg_idxs[k][0], j + neg_idxs[k][1])
                if (Gneg_device_idx not in defect_map_zipped):
                    Gnegs[k, i, j] = accelerator.g_min
                else:
                    # if stuck, mirror
                    Gnegs[k, i, j] = accelerator.read_chip(Gneg_device_idx[0], 1, Gneg_device_idx[1], 1)#accelerator.chip[Gneg_device_idx[0], Gneg_device_idx[1]]
                    # Gnegs[k, i, j] = accelerator._chip[Gneg_device_idx[0], Gneg_device_idx[1]]
    
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
                    optimal_G = quantize_to_nearest(x=optimal_G.numpy(), G_min=G_states[0], d=G_states[1]-G_states[0], n=states)
                    Gposs[k, i, j] = optimal_G
                    if log: print(k, 'adjusted Gpos', Gposs[k, i, j])

                if not Gneg_stuck:
                    if log: print(k, 'adjusting Gneg', Gnegs[k, i, j])

                    Gpos_sum = np.sum(Gposs[:, i, j])
                    Gneg_sum = np.sum(Gnegs[:, i, j])

                    optimal_G = - ((sw_parameter / alpha) * gnorm - Gpos_sum + Gneg_sum - Gnegs[k, i, j])
                    optimal_G = quantize_to_nearest(x=optimal_G.numpy(), G_min=G_states[0], d=G_states[1]-G_states[0], n=states)
                    Gnegs[k, i, j] = optimal_G
                    if log: print(k, 'adjusted Gneg', Gnegs[k, i, j])

            if (log):
                print('adjusted sw parameter', sw_parameter)
                print('adjusted HW parameter', (np.sum(Gposs[:, i, j]) - np.sum(Gnegs[:, i, j])) / gnorm * alpha)

    # convert to pytorch tensors
    Gposs = [torch.from_numpy(Gposs[k]) for k in range(Gposs.shape[0])]
    Gnegs = [torch.from_numpy(Gnegs[k]) for k in range(Gnegs.shape[0])]

    return Gposs, Gnegs