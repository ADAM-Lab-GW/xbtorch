import torch
import sklearn.decomposition as dcmp
import numpy as np
from .base import GenericDecomposition

class FullSVD(GenericDecomposition):

    def __init__(self):
        super().__init__()
        pass

    def decompose(self, input, delta, gradient, group_param_idx):
        U, S, Vh = torch.linalg.svd(gradient, full_matrices=False)
        return U @ torch.diag(S) @ Vh

class TruncatedSVD(GenericDecomposition):

    def __init__(self, rank=1):
        super().__init__()
        self.rank = rank

    def decompose(self, input, delta, gradient, group_param_idx):
        # n_components must be < n_features
        # If full rank, then use `svd_full`
        svd = dcmp.TruncatedSVD(n_components=self.rank)
        svd.fit(gradient)
        # https://stackoverflow.com/questions/31523575/get-u-sigma-v-matrix-from-truncated-svd-in-scikit-learn
        U = svd.transform(gradient).dot(np.linalg.inv(np.diag(svd.singular_values_)))
        SIGMA = svd.singular_values_
        VT = svd.components_
        return torch.Tensor(U @ np.diag(SIGMA) @ VT)