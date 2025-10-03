"""
Predefined neural network architectures for use with XBTorch.

This module provides implementations of standard models such as LeNet-5, VGG,
and RNN/LSTM-based models suitable for image classification and sequential data tasks.

Classes
-------
- :class:`LeNet5` : Classic CNN for small images (e.g., MNIST).
- :class:`VGG` : Deep CNN for image classification tasks (VGG-like).
- :class:`RNN` : Simple RNN model with linear output layer.
- :class:`SimpleLSTM` : LSTM model with linear output for time-series tasks.
"""

import torch
import torch.nn as nn
import xbtorch.nn as xbnn
    
# Define the LeNet-5 model
class LeNet5(nn.Module):
    """
    Classic LeNet-5 CNN architecture for grayscale image classification.

    Architecture:
        - 32C5: Conv2d(1, 32, 5x5), ReLU
        - MP2: MaxPool2d(2x2)
        - 64C5: Conv2d(32, 64, 5x5), ReLU
        - MP2: MaxPool2d(2x2)
        - Flatten
        - 512FC: Linear(1024->512), ReLU
        - 10SSE: Linear(512->10)

    Methods
    -------
    forward(x)
        Forward pass through the network.
    """

    def __init__(self):
        super(LeNet5, self).__init__()
        
        # Define the entire model using nn.Sequential and store it as an attribute
        self.model = nn.Sequential(
            # 32C5: Convolutional layer with 32 filters, 5x5 kernel, input channels = 1 (for grayscale images)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, bias=False),
            nn.ReLU(),
            
            # MP2: Max pooling layer with 2x2 kernel and stride of 2
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 64C5: Convolutional layer with 64 filters, 5x5 kernel
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, bias=False),
            nn.ReLU(),
            
            # MP2: Max pooling layer with 2x2 kernel and stride of 2
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Flatten layer to reshape the tensor
            nn.Flatten(),
            
            # 512FC: Fully connected layer with 512 units
            nn.Linear(64 * 4 * 4, 512, bias=False),
            nn.ReLU(),
            
            # 10SSE: Output layer with 10 units for classification
            nn.Linear(512, 10, bias=False)
        )
        
    def forward(self, x):
        # Forward pass through the Sequential model
        return self.model(x)
    
class VGG(nn.Module):
    """
    VGG-like CNN architecture for image classification.

    Default architecture:
        - 2 × (128C3) - MP2
        - 2 × (256C3) - MP2
        - 2 × (512C3) - MP2
        - Flatten
        - 1024FC
        - num_classes output

    Parameters
    ----------
    num_classes : int
        Number of output classes (default: 10).

    Methods
    -------
    forward(x)
        Forward pass through the network.
    """

    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        # From: https://arxiv.org/pdf/1802.04680
        # 2×(128C3)-MP2-2×(256C3)-MP2-2×(512C3)-MP2-1024FC-10SSE
        self.model = nn.Sequential(
            # Features

            # Group 1
            nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),

            # Group 2
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),

            # Group 3
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            
            # Flatten layer to reshape the tensor
            nn.Flatten(),
            
            # Classifier
            nn.Linear(8192, 1024, bias=False),
            nn.ReLU(),
            nn.Linear(1024, num_classes, bias=False),
        )
        
    def forward(self, x):
        # Forward pass through the Sequential model
        return self.model(x)

class RNN(nn.Module):
    """
    Simple RNN model for sequential data classification.

    Architecture:
        - nn.RNN
        - xbnn.SelectLastStep()  : select last hidden state
        - Linear(hidden_size -> output_size)

    Parameters
    ----------
    input_size : int
        Number of input features per time step.
    hidden_size : int
        Number of hidden units in the RNN.
    num_layers : int
        Number of RNN layers.
    output_size : int
        Number of output classes.

    Methods
    -------
    forward(x)
        Forward pass through the network.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        # if there's a situation where the forward pass can't be re-written as a sequential module directly,
        # we can just include references to layers to be patched here
        self.model = nn.Sequential(
            nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, bias=False),
            xbnn.SelectLastStep(),
            nn.Linear(hidden_size, output_size, bias=False)
        )

    def forward(self, x):
        return self.model(x)

# A Simple LSTM suitable for (simple) time-series datasets
class SimpleLSTM(nn.Module):
    """
    LSTM model suitable for simple time-series or sequence classification tasks.

    Architecture:
        - nn.LSTM
        - xbnn.SelectLastStep() : select last hidden state
        - Linear(hidden_size -> output_size)

    Parameters
    ----------
    input_size : int
        Number of input features per time step.
    hidden_size : int
        Number of hidden units in the LSTM.
    num_layers : int
        Number of LSTM layers.
    output_size : int
        Number of output classes.

    Methods
    -------
    forward(x)
        Forward pass through the network.
    """
    
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleLSTM, self).__init__()

        self.model = nn.Sequential(
            nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=False), # Forward propagate through LSTM; only a single layer means a single cell
            xbnn.SelectLastStep(),
            nn.Linear(hidden_size, output_size, bias=False) # Take the output of the last time step outside the forward
        )

    def forward(self, x):
        # Forward pass through the Sequential model
        return self.model(x)