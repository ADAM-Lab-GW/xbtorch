
"""
Tools for analyzing and visualizing neural network loss landscapes.

This module provides functions to reduce high-dimensional weight spaces
to 2D directions, perturb model weights along those directions, and
compute the corresponding loss grid for visualization or analysis.

Functions
---------
- :func:`reduction_pca` : Reduce weight matrix to 2D using principal component analysis.
- :func:`reduction_random` : Reduce weight matrix to 2D using random orthogonal directions.
- :func:`compute_grid_loss` : Compute loss values on a 2D grid of weight perturbations.
"""

import torch
import torch.optim as optim

from ..nn.utils import train_classifier

import numpy as np
from sklearn.decomposition import PCA


def reduction_pca(matrix, seed=0):
    """
    Perform PCA on a weight matrix to reduce it to two principal directions.

    Parameters
    ----------
    matrix : np.ndarray
        Input weight matrix of shape (num_samples, num_features).
    seed : int, optional
        Random seed for PCA initialization (default: 0).

    Returns
    -------
    tuple
        - path_2d : np.ndarray
            Transformed matrix projected onto the first two principal components.
        - reduced_dirs : np.ndarray
            The principal component directions (2 x num_features).
        - info : dict
            Dictionary containing explained variance ratio: {'explained_vars': array([var1, var2])}.
    """
    pca = PCA(n_components=2, random_state=seed)
    path_2d = pca.fit_transform(matrix)
    reduced_dirs = pca.components_
    return  path_2d, reduced_dirs,  {"explained_vars": pca.explained_variance_ratio_}

def reduction_random(matrix, seed=None):
    """
    Generate two random orthogonal directions for dimensionality reduction.

    Parameters
    ----------
    matrix : np.ndarray
        Input weight matrix of shape (num_samples, num_features).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple
        - None
            Placeholder for 2D path (not computed).
        - reduced_dirs : np.ndarray
            Two random orthonormal directions (2 x num_features).
        - info : dict
            Empty dictionary (no explained variance available).
    """
    if (seed): np.random.seed(seed)

    u_gen = np.random.normal(size=matrix.shape[1])
    u = u_gen / np.linalg.norm(u_gen)
    v_gen = np.random.normal(size=matrix.shape[1])
    v = v_gen / np.linalg.norm(v_gen)
    reduced_dirs = np.array([u, v])

    return None, reduced_dirs,  {}

def compute_grid_loss(model, full_weights, param_name, x_range, y_range, reduced_dirs, data_loader, 
                      criterion, device, epoch, resolution=1, fast=True, log=False):
    """
    Compute the loss landscape on a 2D grid along two given directions in weight space.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model.
    full_weights : dict
        Dictionary of stored weight tensors by layer name and epoch.
        Format: full_weights[layer_name][epoch].
    param_name : str
        Name of the parameter to perturb in the model.
    x_range : tuple(float, float)
        Minimum and maximum scaling along the first direction.
    y_range : tuple(float, float)
        Minimum and maximum scaling along the second direction.
    reduced_dirs : np.ndarray
        2 x num_features array of perturbation directions.
    data_loader : torch.utils.data.DataLoader
        Data loader for evaluating loss.
    criterion : torch.nn.Module
        Loss function.
    device : torch.device
        PyTorch device to perform computations on.
    epoch : int
        Epoch of the stored weights to initialize model.
    resolution : int, optional
        Number of steps per unit interval in the grid (default: 1).
    fast : bool, optional
        If True, perform a faster evaluation during training (default: True).
    log : bool, optional
        If True, print progress logs during computation (default: False).

    Returns
    -------
    tuple
        - xv : np.ndarray
            2D grid of x-axis values.
        - yv : np.ndarray
            2D grid of y-axis values.
        - loss_grid : np.ndarray
            Grid of computed loss values at each perturbation point.
    """
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