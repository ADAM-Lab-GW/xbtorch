import torch.nn as nn
from .decorators import xbtorch

@xbtorch.alter_layer
class Linear(nn.Linear): pass