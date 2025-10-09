# conftest.py
import pytest
import torch
import torch.nn as nn
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import xbtorch

from xbtorch.devices import AnalyticalDevice, AnalyticalIdeal, AnalyticalReal, TabularAnalyticalReal, TabularCompactFeFETKriging
from xbtorch.deployment import SimpleFixedPoint
from xbtorch.patches import xbtorch_model
from xbtorch.decomposition import SBPCA

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size, bias=False),
        )

    def forward(self, x):
        x = x.view(-1, self.input_size)  # Flatten the image
        x = self.model(x)
        return x

# -----------------------
# General Utilities
# -----------------------

@pytest.fixture(scope="session")
def seed():
    """Set global seed for reproducibility."""
    s = 0
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    return s


@pytest.fixture(scope="session")
def device():
    """Return the torch device to use."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# Dataset Fixtures
# -----------------------

@pytest.fixture(scope="session")
def mnist_datasets():
    """Load MNIST datasets once per session."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset


@pytest.fixture
def mnist_loaders(mnist_datasets):
    """Return dataloaders for MNIST."""
    train_dataset, test_dataset = mnist_datasets
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=2)
    return train_loader, test_loader


# -----------------------
# Model Fixtures
# -----------------------

@pytest.fixture
def mlp_model(device):
    """Return a fresh 2-layer MLP patched for XBTorch."""
    input_size = 28*28
    hidden_size = 150
    output_size = 10
    model = SimpleMLP(input_size, hidden_size, output_size).to(device)
    return model

# -----------------------
# XBTorch Accelerator Fixtures
# -----------------------

@pytest.fixture
def simple_fixedpoint_accelerator(device):
    """Return a default SimpleFixedPoint XBTorch accelerator."""
    xb_size = (2500, 2500)
    g_min, g_max = 133, 233
    read_noise, write_noise = 10, 50
    adc_bits = dac_bits = 8
    accelerator = SimpleFixedPoint(
        g_min=g_min,
        g_max=g_max,
        adc_bits=adc_bits,
        dac_bits=dac_bits,
        read_noise=read_noise,
        write_noise=write_noise,
        xb_size=xb_size,
        device=device
    )
    return accelerator

@pytest.fixture
def model_and_acc(mlp_model, device="cpu"):
    """Fixture that returns a patched model and accelerator."""
    acc = SimpleFixedPoint(
        g_min=100,
        g_max=200,
        xb_size=(1000, 1000),
        device=device
    )
    xbtorch.initialize(pytorch_device=device, inference_accelerator=acc)

    model = mlp_model.to(device)
    model = xbtorch_model(model)
    return model, acc, device

@pytest.fixture
def tabular_fefet_device():
    """Return a TabularCompactFeFETKriging device model for gradient decomposition tests."""
    device_model = "tab_fefet_compact_vgs_0.9_sd_5"
    return TabularCompactFeFETKriging(device_model)

@pytest.fixture
def tabular_fefet_acc():
    """Fixture that returns a TabularCompactFeFETKriging accelerator with xbtorch initialized."""
    device = torch.device("cpu")
    xb_device = TabularCompactFeFETKriging("tab_fefet_compact_vgs_0.9_sd_1")
    xbtorch.initialize(
        device_type=xb_device,
        pytorch_device=device,
        weight_range=(-1, +1),
        wage_quantize=True,
        wage_params={"wl_weight": 2, "wl_grad": 8, "wl_activation": 8, "wl_error": 8}
    )
    return device


@pytest.fixture
def simple_mlp_model_tabular():
    """Fixture for a small MLP for HWA/WAGE-style training."""
    class SimpleMLP(torch.nn.Module):
        def __init__(self, input_size=16, hidden_size=8, output_size=4):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, output_size)
            )

        def forward(self, x):
            return self.model(x)

    return SimpleMLP()

# -----------------------
# XBTorch Initialization Fixtures
# -----------------------

@pytest.fixture
def xbtorch_hwa_initialized(simple_fixedpoint_accelerator, device):
    """Initialize XBTorch with a SimpleFixedPoint accelerator."""
    wage_params = {
        "wl_weight": 2,
        "wl_grad": 8,
        "wl_activation": 8,
        "wl_error": 8,
        "rounding_weight": "nearest",
        "rounding_activation": "nearest",
        "rounding_grad": "nearest",
        "rounding_error": "nearest",
    }
    xbtorch.initialize(
        pytorch_device=device,
        inference_accelerator=simple_fixedpoint_accelerator,
        wage_quantize=True,
        wage_params=wage_params
    )
    return xbtorch


@pytest.fixture
def xbtorch_decomp_initialized(tabular_fefet_device, device):
    """Initialize XBTorch for gradient decomposition tests (SBPCA)."""
    decomposition_algorithm = SBPCA(rank=1)
    xbtorch.initialize(
        device_type=tabular_fefet_device,
        pytorch_device=device,
        decomposition_algorithm=decomposition_algorithm
    )
    return xbtorch

# -----------------------
# Fixtures
# -----------------------

@pytest.fixture(params=[AnalyticalIdeal, AnalyticalReal, TabularAnalyticalReal])
def device_preset(request):
    """Fixture to provide different device presets."""
    return request.param()


@pytest.fixture
def custom_analytical_device():
    """Fixture for an AnalyticalDevice with custom parameters."""
    return AnalyticalDevice(
        min_conductance=3e-9,
        max_conductance=3.8e-9,
        d2d_var=0.05,
        c2c_var=0.02,
        nonlinearity_set=5,
        nonlinearity_reset=-5,
        max_level=32
    )


@pytest.fixture
def analytical_real_device():
    """Fixture providing a standard AnalyticalReal device."""
    return AnalyticalReal()


@pytest.fixture
def tabular_analytical_device():
    """Fixture providing a standard TabularAnalyticalReal device."""
    return TabularAnalyticalReal()