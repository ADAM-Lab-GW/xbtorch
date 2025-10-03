"""
Base gradient decomposition module.

This module defines the abstract base class `GenericDecomposition` for
all gradient decomposition strategies in XBTorch, along with a
reference implementation `FullOuterProduct`.

Gradient decomposition methods are used to approximate or compress
gradients during training, which is especially useful for
memristive neural networks or low-rank gradient compression.

Subclasses must implement the `decompose` method to define their
specific decomposition strategy.
"""

import abc
import torch

class GenericDecomposition(metaclass=abc.ABCMeta):
    """
    Abstract base class for gradient decomposition strategies in XBTorch.

    Gradient decomposition methods approximate or compress the gradient
    information during training of memristive neural networks. This is
    particularly useful for reducing communication overhead and memory
    footprint in distributed training, and for minimizing the number of
    device writes in crossbar-based hardware.

    Subclasses must implement the :meth:`decompose` method.

    Notes
    -----
    This class defines the API contract for all gradient decomposition
    methods. Implementations can range from the full outer product
    (baseline) to more advanced low-rank approximations such as
    Streaming Batch PCA, Non-negative Matrix Factorization, or SVD.

    References
    ----------
    For background, see:
    - Huang et al., "Low-rank gradient descent for memory-efficient training of deep in-memory arrays", ACM JETC, 2023.
    - Hoskins et al., "Streaming batch eigenupdates for hardware neural networks", Frontiers in Neuroscience, 2019.

    """

    def __init__(self):
        """
        Initialize the generic decomposition base class.
        """
        pass

    @abc.abstractmethod
    def decompose(self, input, delta, gradient, group_param_idx):
        """
        Decompose the gradient according to the chosen strategy.

        Parameters
        ----------
        input : torch.Tensor
            The input activations for the current layer, with shape
            ``(batch_size, input_dim)``.
        delta : torch.Tensor
            The backpropagated errors (deltas) for the current layer,
            with shape ``(batch_size, output_dim)``.
        gradient : torch.Tensor
            The reference gradient tensor with shape ``(output_dim, input_dim)``,
            used to allocate the output decomposition.
        group_param_idx : int or Any
            Index or identifier for grouping parameters, used by certain
            decomposition methods. Can be ignored if not applicable.

        Returns
        -------
        torch.Tensor
            The decomposed gradient tensor with the same shape as
            ``gradient``.
        """
        pass


class FullOuterProduct(GenericDecomposition):
    """
    Full outer product gradient decomposition.

    This baseline method computes the complete gradient matrix as the
    outer product between deltas (errors) and inputs, aggregated across
    all samples in the batch. It is equivalent to the standard PyTorch
    gradient computation, without scaling by batch size.

    Notes
    -----
    While accurate, this method does not reduce the gradient’s memory
    footprint. It serves as a reference point against which low-rank
    approximations (e.g., PCA, NMF, SVD) can be compared in terms of
    efficiency and robustness on memristive hardware.

    """

    def __init__(self):
        """
        Initialize the full outer product decomposition strategy.
        """
        super().__init__()

    def decompose(self, input, delta, gradient, group_param_idx):
        """
        Compute the full outer product between deltas and inputs.

        Parameters
        ----------
        input : torch.Tensor
            The input activations for the current layer,
            shape ``(batch_size, input_dim)``.
        delta : torch.Tensor
            The backpropagated errors (deltas) for the current layer,
            shape ``(batch_size, output_dim)``.
        gradient : torch.Tensor
            A tensor used to initialize the output with the correct shape
            ``(output_dim, input_dim)``.
        group_param_idx : int or Any
            Unused in this method, included for API consistency.

        Returns
        -------
        torch.Tensor
            The full gradient matrix of shape ``(output_dim, input_dim)``,
            computed as the batch sum of outer products between deltas
            and inputs.
        """
        batch_size = input.shape[0]
        outer_prod = torch.zeros_like(gradient)
        for sample in range(batch_size):
            # <outer<ds, xs>> equivalent to PyTorch's gradient, norm ~ 0 without any scaling by batch size
            # PyTorch does this so that magnitudes of the gradients are scale-invariant of batch size
            outer_prod += torch.outer(delta[sample], input[sample])
        return outer_prod