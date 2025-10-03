"""
Singular Value Decomposition (SVD) decomposition module.

Provides two classes:

- `FullSVD`: Computes the exact full-rank SVD of the gradient,
  serving as a baseline reference.
- `TruncatedSVD`: Computes a low-rank approximation using
  truncated singular values, reducing memory and computation
  while approximating the gradient.

These methods allow benchmarking and comparison against other
low-rank decomposition strategies.
"""

import torch
import sklearn.decomposition as dcmp
import numpy as np
from .base import GenericDecomposition

class FullSVD(GenericDecomposition):
    """
    Full Singular Value Decomposition (SVD) gradient decomposition.

    This method computes the full-rank singular value decomposition of
    the gradient matrix, yielding an exact reconstruction. It serves as
    a reference baseline for evaluating truncated or streaming methods.

    Notes
    -----
    - Uses ``torch.linalg.svd`` for efficient GPU/CPU support.
    - While accurate, the full SVD has high computational cost and does
      not reduce gradient dimensionality.

    References
    ----------
    - Eckart and Young, "The approximation of one matrix by another of lower rank",
      Psychometrika, 1936.
    - Huang et al., "Low-rank gradient descent for memory-efficient training of
      deep in-memory arrays", ACM JETC, 2023.
    """

    def __init__(self):
        """
        Initialize the full SVD decomposition strategy.
        """
        super().__init__()
        pass

    def decompose(self, input, delta, gradient, group_param_idx):
        """
        Compute the full singular value decomposition (SVD) of the gradient.

        Parameters
        ----------
        input : torch.Tensor
            Input activations for the current layer (unused in this method).
        delta : torch.Tensor
            Backpropagated errors for the current layer (unused in this method).
        gradient : torch.Tensor
            Gradient tensor to be decomposed, shape ``(output_dim, input_dim)``.
        group_param_idx : int or Any
            Parameter group index (unused in this method, kept for API consistency).

        Returns
        -------
        torch.Tensor
            Exact reconstruction of the gradient via::

                gradient ≈ U @ diag(S) @ Vh

            where U, S, and Vh are obtained from full SVD.
        """
        U, S, Vh = torch.linalg.svd(gradient, full_matrices=False)
        return U @ torch.diag(S) @ Vh

class TruncatedSVD(GenericDecomposition):
    """
    Truncated Singular Value Decomposition (SVD) gradient decomposition.

    This method approximates the gradient matrix using a rank-limited
    singular value decomposition, providing a compressed low-rank
    representation. It is especially useful for reducing memory and
    communication costs in training memristive neural networks.

    Parameters
    ----------
    rank : int, optional (default=1)
        The number of singular values and vectors to retain. Must be
        less than the number of features (columns of the gradient).

    Notes
    -----
    - Relies on scikit-learn’s :class:`TruncatedSVD` for dimensionality
      reduction.
    - If the gradient rank is equal to or smaller than the input rank,
      consider using :class:`FullSVD` instead.
    - Output reconstruction follows the relationship::

        gradient ≈ U @ diag(Sigma) @ VT

      where U, Sigma, and VT are derived from the truncated factorization.

    References
    ----------
    - Halko et al., "Finding structure with randomness: Probabilistic algorithms
      for constructing approximate matrix decompositions", SIAM Review, 2011.
    - Huang et al., "Low-rank gradient descent for memory-efficient training of
      deep in-memory arrays", ACM JETC, 2023.
    """
    def __init__(self, rank=1):
        """
        Initialize a truncated SVD decomposition instance.

        Parameters
        ----------
        rank : int, optional
            Target rank for the decomposition (default=1).
        """
        super().__init__()
        self.rank = rank

    def decompose(self, input, delta, gradient, group_param_idx):
        """
        Perform truncated SVD-based gradient decomposition.

        Parameters
        ----------
        input : torch.Tensor
            Input activations for the current layer (unused in this method).
        delta : torch.Tensor
            Backpropagated errors for the current layer (unused in this method).
        gradient : torch.Tensor
            Gradient tensor to be decomposed, shape ``(output_dim, input_dim)``.
        group_param_idx : int or Any
            Parameter group index (unused in this method, kept for API consistency).

        Returns
        -------
        torch.Tensor
            Low-rank approximation of the gradient with shape
            ``(output_dim, input_dim)``, reconstructed from truncated
            singular value decomposition.
        """
        # n_components must be < n_features
        svd = dcmp.TruncatedSVD(n_components=self.rank)
        svd.fit(gradient)

        # Extract approximate decomposition
        U = svd.transform(gradient).dot(np.linalg.inv(np.diag(svd.singular_values_)))
        SIGMA = svd.singular_values_
        VT = svd.components_
        return torch.Tensor(U @ np.diag(SIGMA) @ VT)