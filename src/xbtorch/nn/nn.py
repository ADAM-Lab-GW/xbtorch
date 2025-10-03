"""
Custom neural network layers and utilities compatible with XBTorch.

This module provides:
- Decorated versions of standard PyTorch layers for XBTorch compatibility.
- A helper layer to extract the last time-step output from sequence models (RNN/LSTM).
- Sum-of-squared-errors (SSE) loss function suitable for classification tasks.

Classes
-------
- :class:`Linear` : XBTorch-compatible fully connected layer.
- :class:`Conv2d` : XBTorch-compatible 2D convolutional layer.
- :class:`RNN` : XBTorch-compatible RNN layer.
- :class:`LSTM` : XBTorch-compatible LSTM layer.
- :class:`SelectLastStep` : Layer that extracts the last time-step output from sequence models.

Functions
---------
- :func:`SSE(logits, label)` : Computes sum-of-squared-errors loss for one-hot classification.
"""

import torch
import torch.nn as nn
from ..patches import decorators

@decorators.xbtorch_layer
class Linear(nn.Linear): pass

@decorators.xbtorch_layer
class Conv2d(nn.Conv2d): pass

@decorators.xbtorch_layer
class RNN(nn.RNN): pass

@decorators.xbtorch_layer
class LSTM(nn.LSTM): pass

# Custom layer to select the last time step output in time-based models (RNNs, LSTMs)
class SelectLastStep(nn.Module):
    """
    Helper layer to select the output of the last time step from sequence models (RNNs/LSTMs).

    Forward Input
    -------------
    x : tuple
        A tuple of (output, (h_n, c_n)) as returned by RNN/LSTM layers in PyTorch.

    Forward Output
    --------------
    torch.Tensor
        The tensor corresponding to the output of the last time step.
        Shape: (batch_size, hidden_size)
    """
    
    def forward(self, x):
        # x is a tuple (output, (h_n, c_n))
        lstm_out, _ = x
        # Extract the last time step
        return lstm_out[:, -1, :]
    
def SSE(logits, label):
    """
    Compute the sum-of-squared-errors (SSE) loss for one-hot classification tasks.

    Parameters
    ----------
    logits : torch.Tensor
        Predicted outputs from the network. Shape: (batch_size, num_classes)
    label : torch.Tensor
        Ground truth labels. Shape: (batch_size,)

    Returns
    -------
    torch.Tensor
        The scalar SSE loss.
    """
    target = torch.zeros_like(logits)
    target[torch.arange(target.size(0)).long(), label] = 1
    out =  0.5*((logits-target)**2).sum()
    return out

__all__ = ["Linear", "Conv2d", "RNN", "LSTM", "SelectLastStep", "SSE"]