import torch.optim as optim
from .decorators import xbtorch

@xbtorch.alter_optimizer
class SGD(optim.SGD): pass