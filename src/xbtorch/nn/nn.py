import torch.nn as nn
from ..patches import decorators

@decorators.xbtorch_layer
class Linear(nn.Linear): pass