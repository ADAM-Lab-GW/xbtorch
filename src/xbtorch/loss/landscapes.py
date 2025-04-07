import torch
import torch.optim as optim

from ..nn.utils import train_classifier

import numpy as np
from sklearn.decomposition import PCA


def reduction_pca(matrix, seed=0):
    """Perform PCA on the input matrix.

    Returns:
        A tuple of values including the the 2D path,
        the reduced directions, and percentage of variance explained by the
        directions.
    """
    pca = PCA(n_components=2, random_state=seed)
    path_2d = pca.fit_transform(matrix)
    reduced_dirs = pca.components_
    return  path_2d, reduced_dirs,  {"explained_vars": pca.explained_variance_ratio_}

def reduction_random(matrix, seed=None):
    """Random

    Returns:
        The reduced directions, and percentage of variance explained by the
        directions.
    """

    if (seed): np.random.seed(seed)

    u_gen = np.random.normal(size=matrix.shape[1])
    u = u_gen / np.linalg.norm(u_gen)
    v_gen = np.random.normal(size=matrix.shape[1])
    v = v_gen / np.linalg.norm(v_gen)
    reduced_dirs = np.array([u, v])

    return None, reduced_dirs,  {}

def compute_grid_loss(model, full_weights, param_name, x_range, y_range, reduced_dirs, data_loader, criterion, device, epoch, resolution=1, fast=True, log=False):

    step_size = 1 / resolution
    # initiate the grid
    xmin, xmax = x_range
    ymin, ymax = y_range
    x_range = np.arange(xmin, xmax + step_size, step_size)
    y_range = np.arange(ymin, ymax + step_size, step_size)

    xv, yv = np.meshgrid(x_range, y_range, indexing='ij')

    loss_grid = np.zeros((len(x_range), len(y_range)))

    dummy_optimizer = optim.SGD(model.parameters(), lr=0)

    if (log): print("\n")

    for i in range(loss_grid.shape[0]):
        for j in range(loss_grid.shape[1]):

            if (log):
                print(f"Computing loss grid:  ({i+1}, {j+1})/({loss_grid.shape[0]}, {loss_grid.shape[1]})")

            # initialize network with post-training weights
            for name, parameter in model.named_parameters():
                parameter.data = torch.tensor(full_weights[name][epoch], dtype=torch.float32)
                # should this layer be perturbed?
                if (name == param_name):
                    # Calculate the perturbation
                    perturbation = xv[i, j] * reduced_dirs[0] + yv[i, j] * reduced_dirs[1]
                    perturbation = perturbation.reshape(parameter.shape)
                    perturbed_weight = parameter.detach().numpy() + perturbation
                    parameter.data = torch.tensor(perturbed_weight, dtype=torch.float32)

            loss = train_classifier(data_loader, model, criterion, dummy_optimizer, epoch, device=device, log=False, fast=fast)
            loss_grid[i, j] = loss

    return xv, yv, loss_grid