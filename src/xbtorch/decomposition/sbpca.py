import torch
import torch.nn as nn
from .base import GenericDecomposition

class SBPCA(GenericDecomposition):

    def __init__(self, rank=1, sub_batch_size=32):
        '''
        Streaming Batch Principal Component Analysis Decomposition
        https://dl.acm.org/doi/10.1145/3577214
        '''
        super().__init__()
        self.rank = rank
        self.sub_batch_size = sub_batch_size

        self.info = {'X': None, 'sigma': None, 'DELTA': None}

    def decompose(self, input, delta, gradient, group_param_idx):

        if (self.info['X'] is None): 
            for key in self.info.keys(): self.info[key] = {}
            
        if (group_param_idx not in self.info['X'].keys()):
            self.info['X'][group_param_idx] = nn.init.orthogonal_(torch.empty(gradient.shape[1], self.rank))
            self.info['sigma'][group_param_idx] = torch.ones(self.rank)
            self.info['DELTA'][group_param_idx] = nn.init.orthogonal_(torch.empty(gradient.shape[0], self.rank))

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