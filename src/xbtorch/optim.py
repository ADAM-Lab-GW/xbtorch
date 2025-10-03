"""
XBTorch-wrapped optimizers with support for WAGE quantization, device-aware updates,
and decomposition algorithms

This module wraps standard PyTorch optimizers (SGD and Adam) with XBTorch-specific
functionality for:

- Gradient quantization (WAGE)
- Device weight modeling
- Decomposition algorithms
- Weight clipping to specified ranges

Classes
-------
SGD
    XBTorch-wrapped stochastic gradient descent optimizer.
Adam
    XBTorch-wrapped Adam optimizer.
"""

import torch.optim as optim
from .patches import decorators

@decorators.xbtorch_optimizer
class SGD(optim.SGD): pass

@decorators.xbtorch_optimizer
class Adam(optim.Adam): pass