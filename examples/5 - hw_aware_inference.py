'''
Take a pre-trained model
Simulate inference by porting to a mixed-signal prototyping system with some limited precision (ADCs, DACs) and noise profiles
'''

import numpy as np
import random
import time

import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import xbtorch

from xbtorch.devices import AnalyticalIdeal, AnalyticalReal, TabularAnalyticalReal, TabularCompactFeFETKriging, TabularExperimentalFemFETKriging
from xbtorch.patches import xbtorch_model

from xbtorch.devices.utils import test_classifier

from xbtorch.deployment import Daffodil, SimpleFixedPoint, map_random, encode_simple, encode_MAO, compute_error
# from xbtorch.mapping import map_random

import configargparse as argparse
from datetime import datetime
from pathlib import Path

from xbtorch.nn.models import SimpleMLP

from functools import partial

if __name__ == '__main__':

    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    parser = argparse.get_argument_parser(description='XBTorch HWA Inference', default_config_files=['configs/default.ini'])
    parser.add('-c', '--my-config', required=False, is_config_file=True, help='config file path')

    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # XBTorch params
    parser.add_argument('--xb_patch_network', default=False, action='store_true')
    parser.add_argument('--xb_device_model', default=None) # used to figure out where the weights are for TNANO, should be modular
    # parser.add_argument('--experimental_vg', default='2.0') # used to figure out where the weights are for TNANO, should be modular

    parser.add_argument('--fixed_all', default=False, action='store_true')

    parser.add_argument('--in_dir', default='', help='directory where pre-trained weights are (out_dir from train_simple_mlp script)', required=True)

    parser.add_argument('--model', default='mlp')

    args = parser.parse_args()

    print('parsed args', args)

    seed = args.seed
    fixed_all = args.fixed_all

    if (fixed_all):
        seed = 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed) # To control weight update jump table stochasticity

    # Check if CUDA is available and select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.set_default_device(device)

    # We need to specify G_min & G_max
    # This can either be manually provided, or extracted from a device model, as follows

    '''
    device_model = f'tab_femfet_experimental_vg_{args.experimental_vg}'
    xb_device = TabularExperimentalFemFETKriging(device_model)
    g_min = max(xb_device.set_G[0], xb_device.reset_G[0]) * 10**8
    g_max = min(xb_device.set_G[-1], xb_device.reset_G[-1]) * 10**8
    '''

    # Source: https://arxiv.org/pdf/2404.15621
    g_min = 133
    g_max = 233

    # weight_encoding_scheme = encode_simple
    weight_encoding_scheme = encode_MAO
    mapping_scheme = partial(map_random, beta=2)

    output_polling_mode = 'sum' if weight_encoding_scheme == encode_MAO else 'avg'

    print('gmin gmax', g_min, g_max)
    # inference_accelerator = Daffodil(g_min=g_min, g_max=g_max, v_read=0.3) # 300, 450 for Vgs 2.0
    # inference_accelerator = SimpleFixedPoint(adc_bits=12, dac_bits=12, read_noise=0, write_noise=10, stuck_percentage=0.05)
    inference_accelerator = SimpleFixedPoint(g_min=g_min, g_max=g_max, adc_bits=12, dac_bits=12, read_noise=0, write_noise=0, 
                                             stuck_percentage=0.2, 
                                             weight_encoding_scheme=weight_encoding_scheme, 
                                             xb_mapping_scheme=mapping_scheme)

    # todo: weight_range should be selectable from here
    # todo: rename device_type to device
    xbtorch.initialize(
                       pytorch_device=device,
                       inference_accelerator=inference_accelerator,
                       wage_quantize=True # we set this to avoid mismatches between model state dicts, but we do not use wage quantization during the evaluation step
                       )

    # Define transforms to apply to the data
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
        transforms.CenterCrop((18, 18)),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize the image data
    ])

    # Load the MNIST training and test datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders for batching and shuffling
    test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False, generator=torch.Generator(device=device), num_workers=4)

    # Define the model
    input_size = 18 * 18  
    hidden_size = 50
    output_size = 10

    # Model
    if (args.model == 'mlp'): model = SimpleMLP(input_size, hidden_size, output_size).to(device)
    else: raise ValueError("Unspecified model")

    model = xbtorch_model(model)

    import glob

    files = glob.glob(f'{args.in_dir}/statedict_*.pt')

    # infer epochs
    epochs = len(files) - 1

    if (epochs < 1): raise ValueError("Unable to find pre-trained weights") # todo: should just load a state dict

    print('inferencing default HWA trained weights')
    # load pre-trained weights

    model.load_state_dict(torch.load(f'{args.in_dir}/statedict_epoch{epochs}.pt'))

    sw_acc = test_classifier(test_loader, model, device)

    print('inferencing on a XBAR')

    model.xb_eval() # function added if patching successful

    cycles = 10
    accs = np.zeros((cycles, 2))
    for cycle in range(cycles):

        model.initialize_array_mappings(output_polling_mode=output_polling_mode)

        # inference_accelerator.plot_array()
        print('Errors', compute_error(model))
        acc = test_classifier(test_loader, model, device)
        drop = sw_acc - acc
        accs[cycle] = [acc, drop]

        print('Acc', acc, 'Drop from SW', drop)

        inference_accelerator.initialize_chip()

    print(np.average(accs[:, 0]))

        # for layer_idx_loop in layer_idxs:
        #     gneg, gpos, _ = inference_accelerator.get_hw_weights(named_params[layer_idx_loop][1].data)
        #     np.savetxt(f'{args.out_dir}/{device_model}_{args.xb_device_model}_cycle{cycle}_layer{layer_idx_loop}_gneg.txt', gneg)
        #     np.savetxt(f'{args.out_dir}/{device_model}_{args.xb_device_model}_cycle{cycle}_layer{layer_idx_loop}_gpos.txt', gpos)

    # np.savetxt(f'{args.out_dir}/{device_model}_{args.xb_device_model}_acc_drop.txt', accs)