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
from torch.utils.data import DataLoader, ConcatDataset

import xbtorch

from xbtorch.devices import AnalyticalIdeal, AnalyticalReal, TabularAnalyticalReal, TabularCompactFeFETKriging, TabularExperimentalFemFETKriging
from xbtorch.patches import xbtorch_model

from xbtorch.nn.utils import test_classifier

from xbtorch.deployment import SimpleFixedPoint, map_random, encode_simple, encode_MAO, encode_LEA1, encode_LEA2, compute_error

import configargparse as argparse
from datetime import datetime

from xbtorch.nn.models import SimpleMLP, YinYangMLP

from functools import partial
import torch.optim as optim

import glob
import copy

if __name__ == '__main__':

    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    parser = argparse.get_argument_parser(description='XBTorch HWA Inference', default_config_files=['configs/default.ini'])
    parser.add('-c', '--my-config', required=False, is_config_file=True, help='config file path')

    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # XBTorch params
    parser.add_argument('--xb_device_model', default=None) # used to figure out where the weights are for TNANO, should be modular
    # parser.add_argument('--experimental_vg', default='2.0') # used to figure out where the weights are for TNANO, should be modular

    parser.add_argument('--fixed_all', default=False, action='store_true')

    parser.add_argument('--in_dir', default='', help='directory where pre-trained weights are (out_dir from train_simple_mlp script)', required=True)

    parser.add_argument('--model', default='mlp_mnist')

    # Common Params
    parser.add_argument('--weight_encoding_scheme', default='simple')
    parser.add_argument('--xb_mapping_scheme', default='random')
    parser.add_argument('--beta', default=1, type=int, help='Redundancy Ratio')
    parser.add_argument('--stuck_percentage', default=0.0, type=float, help='What % of devices in the simulated chip should have stuck-at-faults?')
    parser.add_argument('--stuck_mode', default='real', type=str, help='Dictates how stuck devices are distributed, options: ideal or real')

    # Params for LEA (Layer Ensemble Averaging)
    parser.add_argument('--alpha', default=1, type=int, help='Out of \beta rows, how many should be used for actual averaging?')

    # Params for FTNNA
    parser.add_argument('--ftnna', default=False, action='store_true', help='Whether or not the FTNNA architecture should be tested')
    parser.add_argument('--num_classifiers', type=int, default=7, 
                        help='Number of classifiers. Default is 7.')
    parser.add_argument('--hamming_distance', type=int, default=3, 
                        help='Hamming distance for code-searching. Default is 3.')
    parser.add_argument('--finetune_epochs', type=int, default=20, 
                        help='Number of epochs for fine-tuning. Default is 20.')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for fine-tune training. Default is 64.')
    
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

    if args.xb_mapping_scheme == 'random':
        mapping_scheme = partial(map_random, beta=args.beta)
    else:
        raise ValueError("Undefined XB mapping scheme")
    

    additional_args = {}

    # output polling modes should be determined automatically based on the encoding scheme that is specified
    # TODO: the output polling mode can simply be computed based on the weight encoding scheme, but we keep this as a todo iterm for later
    if (args.weight_encoding_scheme == 'simple'):
        weight_encoding_scheme = encode_simple
        output_polling_mode = 'avg'
    elif (args.weight_encoding_scheme == 'MAO'):
        weight_encoding_scheme = encode_MAO
        output_polling_mode = 'sum'
    elif (args.weight_encoding_scheme == 'LEA1'):
        weight_encoding_scheme = encode_LEA1
        output_polling_mode = 'reduced_avg'
        additional_args['alpha'] = args.alpha
        additional_args['beta'] = args.beta
    elif (args.weight_encoding_scheme == 'LEA2'):
        weight_encoding_scheme = encode_LEA2
        output_polling_mode = 'reduced_avg'
        additional_args['alpha'] = args.alpha
        additional_args['beta'] = args.beta
    else:
        raise ValueError("Undefined weight encoding scheme")

    # Define the model
    input_size = 28 * 28  
    hidden_size = 150
    output_size = 10

    # load pre-trained weights
    # Model
    if (args.model == 'mlp_mnist'): model = SimpleMLP(input_size, hidden_size, output_size).to(device)
    elif (args.model == 'mlp_yinyang'): model = YinYangMLP().to(device)
    else: raise ValueError("Unspecified model")

    # Load appropriate dataset
    if (args.model == 'mlp_mnist'):
        trained_with_wage = True # we set this to avoid mismatches between model state dicts, but we do not use wage quantization during the evaluation step
        xb_size = (2500, 2500)

        read_noise = 10
        write_noise = 16.66

        # wage_quantize = True # trained with wage quantization, so add this here to
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to tensors
            # transforms.CenterCrop((18, 18)),
            transforms.Normalize((0.1307,), (0.3081,))  # Normalize the image data
        ])

        # Load the MNIST training and test datasets
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        # Create data loaders for batching and shuffling
        test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False, generator=torch.Generator(device=device), num_workers=4)

        if (args.ftnna):
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device=device), num_workers=4)
            from xbtorch.deployment import train_collaborative, test_collaborative, dnn_favorable_searching_code, CollaborativeLoss, add_collaborative_logistic_classifiers

    elif (args.model == 'mlp_yinyang'):
        trained_with_wage = False
        from yinyang_dataset import YinYangDataset
        xb_size = (200, 200)

        # Todo: Update
        read_noise = 0
        write_noise = 0

        yinyang_train = YinYangDataset(size=5000, seed=42)
        yinyang_validation = YinYangDataset(size=1000, seed=41)
        yinyang_test = YinYangDataset(size=1000, seed=40)

        yinyang2_train = YinYangDataset(size=5000, seed=42, offset=1)
        yinyang2_validation = YinYangDataset(size=1000, seed=41, offset=1)
        yinyang2_test = YinYangDataset(size=1000, seed=40, offset=1)

        yinyang_testloader = DataLoader(yinyang_test, batch_size=len(yinyang_test), shuffle=False)
        yinyang2_testloader = DataLoader(yinyang2_test, batch_size=len(yinyang_test), shuffle=False)

        test_loader = [yinyang_testloader, yinyang2_testloader]

        concat_test = ConcatDataset([yinyang_test, yinyang2_test])
        test_loader = DataLoader(concat_test)

        if (args.ftnna): raise ValueError("Not implemented")

    print('gmin gmax', g_min, g_max)
    # inference_accelerator = Daffodil(g_min=g_min, g_max=g_max, v_read=0.3) # 300, 450 for Vgs 2.0
    # inference_accelerator = SimpleFixedPoint(adc_bits=12, dac_bits=12, read_noise=0, write_noise=10, stuck_percentage=0.05)
    inference_accelerator = SimpleFixedPoint(g_min=g_min, g_max=g_max, adc_bits=8, dac_bits=8, 
                                             read_noise=read_noise, 
                                             write_noise=write_noise, 
                                             xb_size=xb_size,
                                             stuck_percentage=args.stuck_percentage, 
                                             stuck_mode=args.stuck_mode,
                                             weight_encoding_scheme=weight_encoding_scheme, 
                                             xb_mapping_scheme=mapping_scheme)

    # todo: weight_range should be selectable from here
    # todo: rename device_type to device
    xbtorch.initialize(
                       pytorch_device=device,
                       inference_accelerator=inference_accelerator,
                       wage_quantize=trained_with_wage 
                       )

    model_dir = args.in_dir
    model = xbtorch_model(model)

    files = glob.glob(f'{model_dir}/statedict_*.pt')
    files.sort()

    # infer epochs
    if (len(files) < 1): raise ValueError("Unable to find pre-trained weights") # todo: should just load a state dict

    model.load_state_dict(torch.load(f'{files[-1]}'))

    print('inferencing default HWA trained weights')
    sw_acc, conf_matrix = test_classifier(test_loader, model, device)

    print('inferencing on a XBAR')
    model.xb_eval() # function added if patching successful        

    cycles = 10
    accs = np.zeros((cycles, 4)) # instead of 4, it should be 2 + num of layers for which error is being computed
    for cycle in range(cycles):

        model.initialize_array_mappings(output_polling_mode=output_polling_mode, additional_args=additional_args, existing_mappings=[])

        # inference_accelerator.plot_array()
        errors = compute_error(model)
        print('Errors', errors)
        acc, _ = test_classifier(test_loader, model, device)
        drop = sw_acc - acc

        if (args.ftnna):
            # Replace softmax classifiers with collaborative logistic classifiers
            class_assignments, codeword_matrix = dnn_favorable_searching_code(conf_matrix, args.num_classifiers, hamming_distance=args.hamming_distance)

            # Let's create a deep copy of the model
            modified_model = copy.deepcopy(model)
            add_collaborative_logistic_classifiers(modified_model, args.num_classifiers)

            collaborative_loss = CollaborativeLoss(codeword_matrix, modified_model)
            optimizer = optim.Adam(modified_model.collaborative_classifier.parameters())

            ftnna_acc = []

            for epoch in range(args.finetune_epochs):
                train_collaborative(modified_model, train_loader, optimizer, collaborative_loss, codeword_matrix, device)
                accuracy = test_collaborative(modified_model, test_loader, codeword_matrix, device)
                ftnna_acc.append(accuracy)
                print(f"Fine-tune Epoch {epoch+1}/{args.finetune_epochs}, Accuracy: {accuracy:.4f}")

            acc = np.max(ftnna_acc)
            print(f"Max Accuracy with Collaborative Logistic Classifiers: {acc:.4f}")
        
        accs[cycle] = [acc, sw_acc - acc, np.min(errors[0]), np.min(errors[1])]

        print('Acc', acc, 'Drop from SW', drop)
        inference_accelerator.initialize_chip()

        if (args.ftnna):
            np.savetxt(f'ftnna_network{args.model}_weightencoding{args.weight_encoding_scheme}_xbmapping{args.xb_mapping_scheme}_beta{args.beta}_stuck{args.stuck_percentage}_stuckmode{args.stuck_mode}_readnoise{round(read_noise, 2)}_writenoise{round(write_noise, 2)}.txt', accs)
        else:
            if (args.weight_encoding_scheme == 'LEA1' or args.weight_encoding_scheme == 'LEA2'):
                np.savetxt(f'network{args.model}_weightencoding{args.weight_encoding_scheme}_xbmapping{args.xb_mapping_scheme}_alpha{args.alpha}_beta{args.beta}_stuck{args.stuck_percentage}_stuckmode{args.stuck_mode}_readnoise{round(read_noise, 2)}_writenoise{round(write_noise, 2)}.txt', accs)
            else:
                np.savetxt(f'network{args.model}_weightencoding{args.weight_encoding_scheme}_xbmapping{args.xb_mapping_scheme}_beta{args.beta}_stuck{args.stuck_percentage}_stuckmode{args.stuck_mode}_readnoise{round(read_noise, 2)}_writenoise{round(write_noise, 2)}.txt', accs)