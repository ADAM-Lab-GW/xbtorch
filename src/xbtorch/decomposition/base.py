import abc
import torch

class GenericDecomposition(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def decompose(self, input, delta, gradient, group_param_idx):
        pass


class FullOuterProduct(GenericDecomposition):

    def __init__(self):
        super().__init__()
        pass

    def decompose(self, input, delta, gradient, group_param_idx):
        batch_size = input.shape[0]
        outer_prod = torch.zeros_like(gradient)
        for sample in range(batch_size):
            # <outer<ds, xs>> equivalent to PyTorch's gradient, norm ~ 0 without any scaling by batch size
            # PyTorch does this so that magnitudes of the gradients are scale-invariant of batch size
            outer_prod += torch.outer(delta[sample], input[sample])
        return outer_prod