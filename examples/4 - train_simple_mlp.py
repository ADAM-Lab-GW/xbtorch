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
from xbtorch.devices import AnalyticalIdeal, AnalyticalReal, TabularAnalyticalReal
from xbtorch.patches import xbtorch_model

from xbtorch.devices.utils import test_classifier, train_classifier, print_num_unique_values

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':

    seed = 0
    fixed_all = True
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
    xb_device = AnalyticalIdeal()
    xb_device = AnalyticalReal()
    xb_device = TabularAnalyticalReal()
    xb_device = None

    # First define number formats used in forward and backward quantization
    # activations 
    weight_range = (-1, +1)

    # WAGE QUANTIZATION
    wage_quantize = True
    wage_params = { 'wl_weight':2, # 2 = ternary weights
                    'wl_grad':8,
                    'wl_activation':8,
                    'wl_error':8,
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
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True, generator=torch.Generator(device=device))
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, generator=torch.Generator(device=device))

    # Define the model
    input_size = 18 * 18  
    hidden_size = 50
    output_size = 10

    model = SimpleMLP(input_size, hidden_size, output_size).to(device)
    model = xbtorch_model(model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    lr = 8
    optimizer = xboptim.SGD(model.parameters(), lr=lr)

    num_epochs = 5
    for epoch in range(num_epochs):
        train_classifier(train_loader, model, criterion, optimizer, epoch, num_epochs, device)
        test_classifier(test_loader, model, device)

    print('Finished Training')

    print_num_unique_values(list(model.named_parameters())[0][1].data)
    print_num_unique_values(list(model.named_parameters())[1][1].data)