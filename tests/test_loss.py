import numpy as np
import torch
import xbtorch.loss as xbloss

def test_loss_reduction_methods_consistency():
    """Reduction via PCA and random projections should both yield valid 2D bases."""
    rng = np.random.default_rng(0)
    M = rng.normal(size=(10, 100))  # mock parameter evolution matrix

    pca_path, reduced_pca, extra_pca = xbloss.reduction_pca(M, seed=0)
    rnd_path, reduced_rnd, extra_rnd = xbloss.reduction_random(M, seed=0)

    dirs_pca = extra_pca.get("reduced_dirs", reduced_pca)
    dirs_rnd = extra_rnd.get("reduced_dirs", reduced_rnd)

    assert dirs_pca.shape[0] == 2
    assert dirs_rnd.shape[0] == 2
    assert "explained_vars" in extra_pca
    assert np.all(extra_pca["explained_vars"] >= 0)

def test_compute_grid_loss_sanity():
    """Mock loss landscape computation should produce finite symmetric grids."""
    class DummyModel(torch.nn.Module):
        def __init__(self, in_features=80, out_features=10):
            super().__init__()
            self.in_features = in_features
            # define the layer immediately so parameters exist
            self.fc = torch.nn.Linear(in_features, out_features, bias=False)

        def forward(self, x):
            x = x.view(-1, self.in_features)  # Flatten the image
            return self.fc(x)

    model = DummyModel(in_features=80, out_features=10)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    # Fake dataloader
    loader = [(torch.randn(8, 80), torch.zeros(8, dtype=torch.long))]

    # Mock weight history
    full_weights = {"fc.weight": np.random.randn(3, 10, 80)}
    dirs = np.random.randn(2, 10 * 80)
    x_range, y_range = (-1, 1), (-1, 1)

    xv, yv, loss_grid = xbloss.compute_grid_loss(
        model, full_weights, "fc.weight",
        x_range, y_range, dirs, loader,
        criterion, device, epoch=0, resolution=1.0, fast=True
    )

    assert xv.shape == yv.shape == loss_grid.shape
    assert np.all(np.isfinite(loss_grid))
    assert loss_grid.ndim == 2

    assert xv.shape == yv.shape == loss_grid.shape
    assert np.all(np.isfinite(loss_grid))
    assert loss_grid.ndim == 2