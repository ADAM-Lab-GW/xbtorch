import torch
import sklearn.decomposition as dcmp
from .base import GenericDecomposition

class NMF(GenericDecomposition):
    def __init__(self, rank=1, streaming=False, max_iters=2000):
        '''
        The streaming argument is based on:
        https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.749811/full
        '''
        super().__init__()
        self.rank = rank
        self.streaming = streaming
        self.max_iters = max_iters

        # estimates are stored internally, used during decomposition if streaming is True.
        self.info = {'Wp': None, 'Hp': None, 'Wn': None, 'Hn': None}

    def decompose(self, input, delta, gradient, group_param_idx=None):

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