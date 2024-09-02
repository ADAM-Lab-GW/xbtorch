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

from xbtorch.nn.utils import test_classifier

from xbtorch.deployment import Daffodil, SimpleFixedPoint, map_random, encode_simple, compute_error, test_committee
# from xbtorch.mapping import map_random

import configargparse as argparse
from datetime import datetime

from xbtorch.nn.models import SimpleMLP

from functools import partial
import torch.optim as optim

import glob
import copy

if __name__ == '__main__':

    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    parser = argparse.get_argument_parser(description='XBTorch HWA Inference', default_config_files=['configs/default.ini'])

    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # XBTorch params
    parser.add_argument('--fixed_all', default=False, action='store_true')
    parser.add_argument('--in_dir', default='', help='directory where pre-trained weights are (out_dir from train_simple_mlp script)', required=True)
    parser.add_argument('--model', default='mlp')

    # Common Params
    parser.add_argument('--weight_encoding_scheme', default='simple')
    parser.add_argument('--xb_mapping_scheme', default='random')
    parser.add_argument('--beta', default=1, type=int, help='Redundancy Ratio')
    parser.add_argument('--stuck_percentage', default=0.0, type=float, help='What % of devices in the simulated chip should have stuck-at-faults?')
    parser.add_argument('--stuck_mode', default='real', type=str, help='Dictates how stuck devices are distributed, options: ideal or real')
    
    args = parser.parse_args()

    print('parsed args', args)

    seed = args.seed
    fixed_all = args.fixed_all

    if (fixed_all):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed) # To control weight update jump table stochasticity

    # Check if CUDA is available and select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.set_default_device(device)

    # Source: https://arxiv.org/pdf/2404.15621
    g_min = 133
    g_max = 233

    # How many unique models will be present in the committee?
    # With committee machines, n unique solutions are mapped once
    # With other schemes, the same solution (in the weight domain) is mapped n times
    if args.xb_mapping_scheme == 'random':
        # committee machine
        mapping_scheme = partial(map_random, beta=1)
    else:
        raise ValueError("Undefined XB mapping scheme")
    
    # output polling modes should be determined automatically based on the encoding scheme that is specified
    if (args.weight_encoding_scheme == 'CM'):
        weight_encoding_scheme = encode_simple
        output_polling_mode = 'avg' # doesn't really matter
    else:
        raise ValueError("Undefined weight encoding scheme")

    read_noise = 10
    write_noise = 16.66

    print('gmin gmax', g_min, g_max)

    inference_accelerator = SimpleFixedPoint(g_min=g_min, g_max=g_max, adc_bits=8, dac_bits=8, 
                                             read_noise=read_noise, 
                                             write_noise=write_noise, 
                                             stuck_percentage=args.stuck_percentage, 
                                             stuck_mode=args.stuck_mode,
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
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize the image data
    ])

    # Load the MNIST training and test datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders for batching and shuffling
    test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False, generator=torch.Generator(device=device), num_workers=4)

    # Define the model
    input_size = 28 * 28  
    hidden_size = 150
    output_size = 10

    # load pre-trained weights
    # Model
    all_models = []

    # gather the \beta models
    for i in range(args.beta):

        if (args.model == 'mlp'): model = SimpleMLP(input_size, hidden_size, output_size).to(device)
        else: raise ValueError("Unspecified model")

        model_dir = args.in_dir.replace('{i}', str(i))
        model = xbtorch_model(model)

        files = glob.glob(f'{model_dir}/statedict_*.pt')

        # infer epochs
        epochs = len(files) - 1
        if (epochs < 1): raise ValueError("Unable to find pre-trained weights") # todo: should just load a state dict

        model.load_state_dict(torch.load(f'{model_dir}/statedict_epoch{epochs}.pt'))

        all_models.append(model)

        print('inferencing default HWA trained weights')
        sw_acc, conf_matrix = test_classifier(test_loader, model, device)

    # let's also test the complete committee network (still in the SW domain)
    test_committee(test_loader, all_models, device)

    cycles = 10
    accs = np.zeros((cycles, 4)) # instead of 4, it should be 2 + num of layers for which error is being computed
    for cycle in range(cycles):

        # Finally, let's perform inference on defective xbars
        # First, map all solutions to the crossbar
        
        # providing this (used as a reference internally) ensures that subsequent models do not overlap with existing model mappings on the xbar
        existing_mappings = []
        errors_all = []
        for idx, model in enumerate(all_models):
            model.xb_eval() # function added if patching successful        
            model.initialize_array_mappings(output_polling_mode=output_polling_mode, existing_mappings=existing_mappings)

        # Let's visualize the array
        # inference_accelerator.plot_array()

            errors = compute_error(model)
            errors_all.append(errors)
        
        errors_all = np.average(errors_all, axis=0)
        print('Errors', errors)
        acc  = test_committee(test_loader, all_models, device)
        drop = sw_acc - acc
        
        accs[cycle] = [acc, sw_acc - acc, np.min(errors_all[0]), np.min(errors_all[1])]

        print('Acc', acc, 'Drop from SW', drop)
        # re-initialize for next cycle
        inference_accelerator.initialize_chip()

        np.savetxt(f'network{args.model}_weightencoding{args.weight_encoding_scheme}_xbmapping{args.xb_mapping_scheme}_beta{args.beta}_stuck{args.stuck_percentage}_stuckmode{args.stuck_mode}_readnoise{round(read_noise, 2)}_writenoise{round(write_noise, 2)}.txt', accs)