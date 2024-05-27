import torch.optim as optim
from .patches import decorators

@decorators.xbtorch_optimizer
class SGD(optim.SGD): pass

@decorators.xbtorch_optimizer
class Adam(optim.Adam): pass