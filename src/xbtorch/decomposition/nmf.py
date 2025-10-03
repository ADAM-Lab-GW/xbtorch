"""
Non-negative Matrix Factorization (NMF) decomposition module.

Provides the `NMF` class, which implements gradient decomposition
using a low-rank non-negative matrix factorization. Supports both
standard batch and streaming modes.

This method separates positive and negative components of the
gradient, reducing memory usage and communication costs while
maintaining reasonable approximation accuracy.
"""

import torch
import sklearn.decomposition as dcmp
from .base import GenericDecomposition

class NMF(GenericDecomposition):
    """
    Non-negative Matrix Factorization (NMF) gradient decomposition.

    This method approximates the full gradient matrix as a low-rank
    factorization into non-negative components, separating positive
    and negative contributions. The decomposition reduces the memory
    and communication cost of training while potentially reducing
    the number of device writes in memristive crossbar arrays.

    XBTorch supports two operation modes:
    
    - **Standard mode:** Each batch is decomposed independently.
    - **Streaming mode:** Estimates from previous batches are reused
      (naive streaming), as described in [Hoskins et al., 2021].
      This can improve stability across batches but is approximate.

    Parameters
    ----------
    rank : int, optional (default=1)
        The target rank for the low-rank factorization. Smaller ranks
        yield greater compression but less fidelity.
    streaming : bool, optional (default=False)
        If True, reuses decomposition factors across batches for
        incremental ("streaming") updates. Requires `group_param_idx`
        in :meth:`decompose`.
    max_iters : int, optional (default=2000)
        Maximum number of iterations for the sklearn NMF solver.

    Attributes
    ----------
    info : dict
        Stores intermediate decomposition estimates (Wp, Hp, Wn, Hn)
        for reuse in streaming mode. Keys are grouped by parameter
        index.

    Notes
    -----
    - Positive and negative parts of the gradient are decomposed
      separately to respect the non-negativity constraint of NMF.
    - Streaming mode here is "naive": it re-initializes new NMF runs
      with previous factor estimates but still processes the full
      gradient each time.

    References
    ----------
    - Hoskins et al., "Streaming batch eigenupdates for hardware neural networks",
      Frontiers in Neuroscience, 2019.
    - Vogels et al., "PowerSGD: Practical low-rank gradient compression for
      distributed optimization", NeurIPS 2019.
    - Huang et al., "Low-rank gradient descent for memory-efficient training
      of deep in-memory arrays", ACM JETC, 2023.
    """

    def __init__(self, rank=1, streaming=False, max_iters=2000):
        """
        Initialize an NMF gradient decomposition instance.

        Parameters
        ----------
        rank : int, optional
            Target rank for the decomposition (default=1).
        streaming : bool, optional
            Whether to enable streaming updates across batches
            (default=False).
        max_iters : int, optional
            Maximum number of iterations for sklearn's NMF solver
            (default=2000).
        """
        super().__init__()
        self.rank = rank
        self.streaming = streaming
        self.max_iters = max_iters

        # estimates are stored internally, used during decomposition if streaming is True.
        self.info = {'Wp': None, 'Hp': None, 'Wn': None, 'Hn': None}

    def decompose(self, input, delta, gradient, group_param_idx=None):
        """
        Perform NMF-based gradient decomposition.

        Parameters
        ----------
        input : torch.Tensor
            Input activations for the current layer, shape
            ``(batch_size, input_dim)``.
        delta : torch.Tensor
            Backpropagated errors for the current layer, shape
            ``(batch_size, output_dim)``.
        gradient : torch.Tensor
            Gradient matrix to be decomposed, shape
            ``(output_dim, input_dim)``.
        group_param_idx : int, optional
            Identifier for parameter grouping. Required in streaming
            mode to manage reuse of factors.

        Returns
        -------
        torch.Tensor
            Low-rank approximation of the gradient with shape
            ``(output_dim, input_dim)``, reconstructed as::

                approx_grad = (Wp @ Hp) - (Wn @ Hn)

            where `Wp, Hp` are the positive factorization components,
            and `Wn, Hn` are the negative components (flipped positive
            before NMF).

        Raises
        ------
        RuntimeError
            If streaming is enabled but `group_param_idx` is not
            provided.

        Notes
        -----
        - Positive and negative parts of the gradient are decomposed
          separately, with the negative part flipped to satisfy
          non-negativity.
        - In streaming mode, factor estimates from previous calls are
          reused as custom initializations for subsequent decompositions.
        """
        if (self.streaming and group_param_idx is None):
            raise RuntimeError("group_param index must be provided for streaming NMF.")

        # Separate into +ve, and -ve counter parts (ReLU operation)
        outer_prod_p = gradient.clone()
        outer_prod_n = gradient.clone()

        outer_prod_p[outer_prod_p < 0] = 0 # will only contain +ve entries
        outer_prod_n[outer_prod_n > 0] = 0 # will only contain -ve entries
        outer_prod_n *= -1 # flip to make +ve in order for NMF to work

        # Apply NMF
        if (not self.streaming or self.info['Wp'] is None or (group_param_idx not in self.info['Wp'])):
            # init randomly
            nmfp = dcmp.NMF(n_components=self.rank, init='random', max_iter=self.max_iters)
            Wp = nmfp.fit_transform(outer_prod_p)
            Hp = nmfp.components_

            nmfn = dcmp.NMF(n_components=self.rank, init='random', max_iter=self.max_iters)
            Wn = nmfn.fit_transform(outer_prod_n)
            Hn = nmfn.components_

        else:
            # This isn't really proper streaming, since the full outerproduct gradient is used
            # Naive streaming
            # Stream current best estimates to the next batch
            # init based on previous "best" estimates
            # See paper, supplementary material for details
            nmfp = dcmp.NMF(n_components=self.rank, init='custom', max_iter=self.max_iters)
            Wp = nmfp.fit_transform(outer_prod_p, W=self.info['Wp'][group_param_idx], H=self.info['Hp'][group_param_idx])
            Hp = nmfp.components_

            nmfn = dcmp.NMF(n_components=self.rank, init='custom', max_iter=self.max_iters)
            Wn = nmfn.fit_transform(outer_prod_n, W=self.info['Wn'][group_param_idx], H=self.info['Hn'][group_param_idx])
            Hn = nmfn.components_

        # update estimates
        if (self.info['Wp'] is None): 
            for key in self.info.keys(): self.info[key] = {}

        elif (group_param_idx not in self.info['Wp']):
            self.info['Wp'][group_param_idx] = Wp
            self.info['Hp'][group_param_idx] = Hp
            self.info['Wn'][group_param_idx] = Wn
            self.info['Hn'][group_param_idx] = Hn

        return torch.Tensor((Wp @ Hp) - (Wn @ Hn)) 