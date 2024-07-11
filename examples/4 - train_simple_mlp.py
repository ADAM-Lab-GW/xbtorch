import numpy as np
import random
import time

import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import xbtorch
import xbtorch.optim as xboptim
from xbtorch.decomposition import FullSVD, TruncatedSVD, NMF, SBPCA, FullOuterProduct
from xbtorch.devices import AnalyticalIdeal, AnalyticalReal, TabularAnalyticalReal, TabularCompactFeFETKriging
from xbtorch.patches import xbtorch_model
from xbtorch.nn.models import SimpleMLP

from xbtorch.devices.utils import test_classifier, train_classifier, print_num_unique_values

import configargparse as argparse
from datetime import datetime
from pathlib import Path

def compute_loss_at_grid_point(args):
    i, j, reduced_dirs, model, layer_idx, full_weights, train_loader, criterion, device, epoch = args
    
    # initialize network with post-training weights
    for layer_idx_loop in layer_idxs:
        named_params[layer_idx_loop][1].data = torch.tensor(full_weights[layer_idx_loop][epoch], dtype=torch.float32)

    # Calculate the perturbation
    perturbation = xv[i, j] * reduced_dirs[0] + yv[i, j] * reduced_dirs[1]
    perturbation = perturbation.reshape(named_params[layer_idx][1].shape)
    perturbed_weight = named_params[layer_idx][1].detach().numpy() + perturbation

    named_params[layer_idx][1].data = torch.tensor(perturbed_weight, dtype=torch.float32)

    # acc = test_classifier(train_loader, model, device) # sanity check; can remove perturbations to ensure accuracy matches expected performance

    # Initialize the total loss for the grid point
    total_loss = 0.0
    total_samples = 0
    correct = 0

    # Iterate over the entire dataset
    for images, labels in train_loader:
        images = torch.flatten(images, start_dim=1).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)

        # Accumulate the total loss and sample count
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        correct += (predicted == labels).sum().item()

        break # remove, only for makign it faster temporarily

    acc = 100 * correct / total_samples
    print('acc', acc)
    
    # Average the total loss over the number of samples
    avg_loss = total_loss / total_samples
    print('returning', i, j, avg_loss)
    return (i, j, avg_loss)

if __name__ == '__main__':

    # TODO: Convert loss landscapes to utilize updated model saving (full model save instead of .txt files)

    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    parser = argparse.get_argument_parser(description='XBTorch Network Training', default_config_files=['configs/default.ini'])
    parser.add('-c', '--my-config', required=False, is_config_file=True, help='config file path')

    # Network level/General params
    parser.add_argument('--model', default='mlp')

    parser.add_argument('--batch_size', type=int, default=4096, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=8, help='Network learning rate')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    # WAGE Quantization params
    parser.add_argument('--wage_quantize', default=True, action='store_true')
    parser.add_argument('--wl_weight', type = int, default=2)
    parser.add_argument('--wl_activation', type = int, default=8)
    parser.add_argument('--wl_grad', type = int, default=8)
    parser.add_argument('--wl_error', type = int, default=8)

    # XBTorch params
    parser.add_argument('--xb_patch_network', default=False, action='store_true')
    parser.add_argument('--xb_device_model', default=None)

    # General logging
    parser.add_argument('--log_data', default = False, action='store_true')
    parser.add_argument('--out_dir', default=f'sim_results_{time.strftime("%Y%m%d-%H%M%S")}')
    parser.add_argument('--out_file', default='')

    parser.add_argument('--log_loss_landscape', default = False, action='store_true')
    parser.add_argument('--loss_reduction_method', default = 'pca')

    # Tasks for FeFET Jump Tables IEEE NANO - NANOARCH 2024 follow up
    parser.add_argument('--stuck_percent_all', default=0.0, type=float)
    parser.add_argument('--fixed_all', default=False, action='store_true')

    args = parser.parse_args()

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

    # Decomposition Algorithms
    decomposition_algorithm = FullSVD()
    decomposition_algorithm = TruncatedSVD(rank=2)
    decomposition_algorithm = NMF(rank=2, streaming=False)
    decomposition_algorithm = NMF(rank=2, streaming=True)
    decomposition_algorithm = SBPCA(rank=1)
    decomposition_algorithm = FullOuterProduct()
    decomposition_algorithm = None

    # Device Types
    if (not args.xb_device_model): xb_device = None
    elif (args.xb_device_model == 'an_ideal'): xb_device = AnalyticalIdeal()
    elif (args.xb_device_model == 'an_real'): xb_device = AnalyticalReal()
    elif (args.xb_device_model == 'tab_an_real'): xb_device = TabularAnalyticalReal()
    # For TNANO - shouldn't be a required part for actual framework
    elif ('tab_fefet_compact' in args.xb_device_model): xb_device = TabularCompactFeFETKriging(args.xb_device_model)
    
    print('using device model', xb_device)

    if (args.out_file != ''): exp_name = args.out_file
    else: exp_name = 'b-' +str(args.batch_size) +'-lr-' + str(args.lr)

    if (args.out_dir != ''):
        exp_name = args.out_dir + exp_name
        Path(args.out_dir).mkdir(parents=False, exist_ok=True)

    # First define number formats used in forward and backward quantization
    # activations 
    weight_range = (-1, +1)

    # WAGE QUANTIZATION
    wage_quantize = args.wage_quantize
    wage_params = { 'wl_weight': args.wl_weight, # 2 = ternary weights
                    'wl_grad': args.wl_grad,
                    'wl_activation': args.wl_activation,
                    'wl_error': args.wl_error,
                    'rounding_weight' : 'nearest',
                    'rounding_activation' : 'nearest',
                    'rounding_grad' : 'nearest',
                    'rounding_error' : 'nearest',
                   }

    # todo: weight_range should be selectable from here
    # todo: rename device_type to device
    xbtorch.initialize(decomposition_algorithm=decomposition_algorithm, 
                       device_type=xb_device,
                       pytorch_device=device,
                       weight_range=weight_range,
                       wage_quantize=wage_quantize,
                       wage_params=wage_params
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device=device), num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False, generator=torch.Generator(device=device), num_workers=4)

    # Define the model
    input_size = 18 * 18  
    hidden_size = 50
    output_size = 10

    # Model
    if (args.model == 'mlp'): model = SimpleMLP(input_size, hidden_size, output_size).to(device)
    else: raise ValueError("Unspecified model")

    if (args.xb_patch_network): model = xbtorch_model(model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    lr = args.lr
    optimizer = xboptim.SGD(model.parameters(), lr=lr)

    start = time.time()

    for epoch in range(0, args.epochs):

        if (args.log_data): torch.save(model.state_dict(), f'{args.out_dir}/statedict_epoch{epoch}.pt')

        loss = train_classifier(train_loader, model, criterion, optimizer, epoch, args.epochs, device)
        acc = test_classifier(test_loader, model, device)

        elapsed_time = time.time() - start

        # print_num_unique_values(list(model.named_parameters())[0][1].data)
        # print_num_unique_values(list(model.named_parameters())[1][1].data)

    if (args.log_data): 
        torch.save(model.state_dict(), f'{args.out_dir}/statedict_epoch{epoch+1}.pt')
        import json
        with open(f'{args.out_dir}/params.json', 'w') as fp:
            json.dump(args.__dict__, fp)


    print('Finished Training')

    if (args.log_loss_landscape and args.epochs == 0):
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        import glob

        if (args.xb_device_model and 'fefet_compact' in args.xb_device_model):
            name = args.xb_device_model.split('_')
            vgs, sd = name[-3], name[-1]
            files = glob.glob(f'{args.out_dir}/*vgs_{vgs}_sd_{sd}-*_param_0.txt')
        else:
            vgs, sd = None, None
            files = glob.glob(f'{args.out_dir}/*_param_0.txt')
        # infer epochs

        epochs = len(files)

        # single layer
        layer_idxs = [0, 1]
        layer_idx = 0 # idx to named_params
        landscape_epoch = 3 # which epoch weights should be used for original weights?

        # concatenate + flatten weights
        full_weights = {}
        for layer_idx_loop in layer_idxs:
            full_weights[layer_idx_loop] = np.zeros((epochs, *named_params[layer_idx_loop][1].shape))

        for epoch in range(epochs):
            for layer_idx_loop in layer_idxs:

                if (vgs):
                    files = glob.glob(f'{args.out_dir}/*vgs_{vgs}_sd_{sd}-*_{epoch}_param_{layer_idx_loop}.txt')
                else:
                    files = glob.glob(f'{args.out_dir}/*_{epoch}_param_{layer_idx_loop}.txt')
            
                assert len(files) == 1
                file = files[0]
                data = np.loadtxt(file)
                full_weights[layer_idx_loop][epoch] = data

        '''
        Constructing matrix M = [θ0−θn; · · · ; θn−1−θn] for layer at layer_idx
        '''

        M = []

        for epoch in range(epochs - 1):
            diff = full_weights[layer_idx][epoch] - full_weights[layer_idx][-1]
            M.append(diff.flatten())

        M = np.array(M)

        reduction_method = args.loss_reduction_method
        if (reduction_method == 'pca'):
            pca = PCA(n_components=2, random_state=args.seed)
            components = pca.fit_transform(M)
            reduced_dirs = pca.components_ # unit vectors by default
            explained_vars = pca.explained_variance_ratio_ * 100
        elif(reduction_method == 'random'):
            # use the following to generate random vectors
            load = True
            if (load):
                reduced_dirs = np.loadtxt(f'random_vectors_{full_weights[layer_idx].shape[1] * full_weights[layer_idx].shape[2]}.txt')
            else:
                u_gen = np.random.normal(size=full_weights[layer_idx].shape[1] * full_weights[layer_idx].shape[2])
                u = u_gen / np.linalg.norm(u_gen)
                v_gen = np.random.normal(size=full_weights[layer_idx].shape[1] * full_weights[layer_idx].shape[2])
                v = v_gen / np.linalg.norm(v_gen)
                reduced_dirs = np.array([u, v])

                np.savetxt(f'random_vectors_{full_weights[layer_idx].shape[1] * full_weights[layer_idx].shape[2]}.txt', reduced_dirs)
                exit()

        print(np.linalg.norm(reduced_dirs[0, :]))
        print(np.linalg.norm(reduced_dirs[1, :]))

        # assert (np.linalg.norm(reduced_dirs[0, :])) == 1.0
        # assert (np.linalg.norm(reduced_dirs[1, :])) == 1.0

        # Define a grid in the space of the principal components

        resolution = 1 # increase to decrease step size proportionally
        step_size = 1 / resolution

        xmin, xmax = -5, 5
        ymin, ymax = -5, 5
        x_range = np.arange(xmin, xmax + step_size, step_size)
        y_range = np.arange(ymin, ymax + step_size, step_size)

        if (landscape_epoch == -1):
            loss_filename = f'{args.out_dir}/{args.xb_device_model}_loss_landscape_x_{xmin}_{xmax}_y_{ymin}_{ymax}_res{round(resolution, 2)}_reduction{reduction_method}.txt'
        else:
            loss_filename = f'{args.out_dir}/{args.xb_device_model}_loss_landscape_x_{xmin}_{xmax}_y_{ymin}_{ymax}_res{round(resolution, 2)}_reduction{reduction_method}_epoch{landscape_epoch}.txt'

        xv, yv = np.meshgrid(x_range, y_range, indexing='ij')

        if (Path(loss_filename).exists()):
            print('file exists, skipping computation')
            loss_values = np.loadtxt(loss_filename)

        else:
            # Initialize a matrix to store the loss values
            loss_values = np.zeros((len(x_range), len(y_range)))
            train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=False)#, generator=torch.Generator(device=device))

            # Prepare the arguments for multiprocessing
            # Multiprocessing - experimental
            args_list = [(i, j, reduced_dirs, model, layer_idx, full_weights, train_loader, criterion, device, landscape_epoch) for i in range(len(x_range)) for j in range(len(y_range))]

            with ThreadPoolExecutor(max_workers=1) as executor:
                # valid only if used with a single worker
                results = list(executor.map(compute_loss_at_grid_point, args_list))

            for i, j, avg_loss in results:
                loss_values[i, j] = avg_loss

            np.savetxt(loss_filename, loss_values)

        print(loss_values)

        # Plot the loss landscape
        plt.figure()
        CS = plt.contour(xv, yv, loss_values, levels=20, alpha=0.9)#, cmap="YlGnBu")

        plt.clabel(CS, inline=True, fontsize=8)

        path_projection = M.dot(reduced_dirs.T) / np.linalg.norm(reduced_dirs)

        if (reduction_method == 'pca'):
            np.savetxt(f'{args.out_dir}/{args.xb_device_model}_loss_landscape_info.txt', pca.explained_variance_ratio_ * 100)

        np.savetxt(f'{args.out_dir}/{args.xb_device_model}_loss_landscape_projection{reduction_method}.txt', path_projection)

        # Plot the path of the weights
        start_idx = 1
        plt.plot(path_projection[:, 0][start_idx:], path_projection[:, 1][start_idx:], color='red', linewidth=1)

        min_loss_point = path_projection[-1]

        plt.plot(
            min_loss_point[0], min_loss_point[1], "ro", label="target local minimum"
        )

        # plt.colorbar(label='Loss')
        if (reduction_method == 'pca'):
            plt.xlabel(f'1st Principal Component: {round(explained_vars[0], 2)} %')
            plt.ylabel(f'2nd Principal Component: {round(explained_vars[1], 2)} %')
        elif (reduction_method == 'random'):
            plt.xlabel(f'1st Random Component')
            plt.ylabel(f'2nd Random Component')

        plt.title('Loss Landscape')
        plt.show()

