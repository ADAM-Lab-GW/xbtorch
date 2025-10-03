"""
Streaming Batch PCA (SBPCA) decomposition module.

Implements the `SBPCA` class, which performs an incremental,
streaming principal component analysis on gradients. This allows
low-rank approximations without computing the full gradient matrix,
reducing memory and computational overhead.

Useful for memristive neural networks to reduce device writes and
improve training efficiency.
"""

import torch
import torch.nn as nn
from .base import GenericDecomposition

class SBPCA(GenericDecomposition):
    """
    Streaming Batch Principal Component Analysis (SBPCA) gradient decomposition.

    This method applies an incremental, streaming variant of PCA to
    approximate the full gradient matrix with a low-rank factorization.
    By operating on sub-batches of data, SBPCA avoids forming the
    complete gradient explicitly, thereby reducing memory and compute
    costs.

    SBPCA is particularly useful in memristive neural networks where
    gradient compression reduces the number of device writes, mitigating
    endurance limitations and improving training efficiency.

    Parameters
    ----------
    rank : int, optional (default=1)
        Target rank for the decomposition. Smaller values yield
        greater compression but less accurate approximations.
    sub_batch_size : int, optional (default=32)
        Size of the sub-batches used for incremental updates during
        streaming PCA.

    Attributes
    ----------
    info : dict
        Stores intermediate estimates for each parameter group:
        - ``X``: right singular vectors
        - ``DELTA``: left singular vectors
        - ``sigma``: singular values

    Notes
    -----
    - SBPCA incrementally updates low-rank approximations across
      sub-batches using QR decompositions for numerical stability.
    - This implementation follows the methodology in:
      Huang et al., "Low-rank gradient descent for memory-efficient
      training of deep in-memory arrays", ACM JETC 2023.

    References
    ----------
    - Huang et al., "Low-rank gradient descent for memory-efficient
      training of deep in-memory arrays", ACM JETC, 2023.
    - Hoskins et al., "Streaming batch eigenupdates for hardware
      neural networks", Frontiers in Neuroscience, 2019.
    """

    def __init__(self, rank=1, sub_batch_size=32):
        """
        Initialize an SBPCA decomposition instance.

        Parameters
        ----------
        rank : int, optional
            Target rank for the decomposition (default=1).
        sub_batch_size : int, optional
            Sub-batch size for incremental updates (default=32).
        """
        super().__init__()
        self.rank = rank
        self.sub_batch_size = sub_batch_size

        self.info = {'X': None, 'sigma': None, 'DELTA': None}

    def decompose(self, input, delta, gradient, group_param_idx):
        """
        Perform SBPCA-based gradient decomposition.

        Parameters
        ----------
        input : torch.Tensor
            Input activations for the current layer, shape
            ``(batch_size, input_dim)``.
        delta : torch.Tensor
            Backpropagated errors for the current layer, shape
            ``(batch_size, output_dim)``.
        gradient : torch.Tensor
            Gradient tensor (used for shape and initialization),
            shape ``(output_dim, input_dim)``.
        group_param_idx : int
            Identifier for parameter grouping, used to track state
            across multiple calls.

        Returns
        -------
        torch.Tensor
            Low-rank approximation of the gradient with shape
            ``(output_dim, input_dim)``, reconstructed from::

                approx_grad = DELTA @ diag(sigma) @ X^T

            scaled by the batch size.

        Notes
        -----
        - If state for `group_param_idx` does not exist, orthogonal
          initializations are created for X and DELTA.
        - QR decompositions are used to maintain orthogonality of
          factors during updates.
        - For rank > 1, only the leading `rank` components are kept.
        """
        if (self.info['X'] is None): 
            for key in self.info.keys(): self.info[key] = {}
            
        if (group_param_idx not in self.info['X'].keys()):
            self.info['X'][group_param_idx] = nn.init.orthogonal_(torch.empty(gradient.shape[1], self.rank)).to(gradient.device)
            self.info['sigma'][group_param_idx] = torch.ones(self.rank).to(gradient.device)
            self.info['DELTA'][group_param_idx] = nn.init.orthogonal_(torch.empty(gradient.shape[0], self.rank)).to(gradient.device)

        batch_size = input.shape[0]

        xs = input
        ds = delta
        for i in range(batch_size // self.sub_batch_size):
            y = torch.mm(ds[range(i * self.sub_batch_size, (i + 1) * self.sub_batch_size)], self.info['DELTA'][group_param_idx]) / ((i + 2) * self.sub_batch_size)
            self.info['X'][group_param_idx] = self.info['X'][group_param_idx] * (i + 1) * self.sub_batch_size / ((i + 2) * self.sub_batch_size) + torch.mm(torch.transpose(xs[range(i * self.sub_batch_size, (i + 1) * self.sub_batch_size)], 0 ,1), y) / self.info['sigma'][group_param_idx]
            Q, R = torch.linalg.qr(self.info['X'][group_param_idx])
            self.info['X'][group_param_idx] = torch.mm(Q,torch.diag(torch.diag(torch.sign(R))))
            z = torch.mm(xs[range(i * self.sub_batch_size, (i + 1) * self.sub_batch_size)], self.info['X'][group_param_idx]) / ((i + 2) * self.sub_batch_size)
            self.info['DELTA'][group_param_idx] = self.info['DELTA'][group_param_idx] * (i + 1) * self.sub_batch_size / ((i + 2) * self.sub_batch_size) + torch.mm(torch.transpose(ds[range(i * self.sub_batch_size, (i + 1) * self.sub_batch_size)], 0, 1), z) / self.info['sigma'][group_param_idx]
            Q, R = torch.linalg.qr(self.info['DELTA'][group_param_idx])
            self.info['DELTA'][group_param_idx] = torch.mm(Q,torch.diag(torch.diag(torch.sign(R))))
            self.info['sigma'][group_param_idx] = self.info['sigma'][group_param_idx] * (i + 1) * self.sub_batch_size / ((i + 2) * self.sub_batch_size) + torch.sum(torch.mm(ds[range(i * self.sub_batch_size, (i + 1) * self.sub_batch_size)], self.info['DELTA'][group_param_idx]) * torch.mm(xs[range(i * self.sub_batch_size, (i + 1) * self.sub_batch_size)], self.info['X'][group_param_idx]), 0) / ((i + 2) * self.sub_batch_size)
        
        # Construct the final self.info['ork'][group_param_idx] layer gradient
        k = self.rank
        if self.info['sigma'][group_param_idx].shape[0] == 1 or self.info['sigma'][group_param_idx].shape[0] == k:
            return torch.mm(torch.mm(self.info['DELTA'][group_param_idx], torch.diag(self.info['sigma'][group_param_idx])), torch.transpose(self.info['X'][group_param_idx], 0, 1)) * batch_size
        else:
            return torch.mm(torch.mm(self.info['DELTA'][group_param_idx][:, : k], torch.diag(self.info['sigma'][group_param_idx][: k])), torch.transpose(self.info['X'][group_param_idx][:, : k], 0, 1)) * batch_size