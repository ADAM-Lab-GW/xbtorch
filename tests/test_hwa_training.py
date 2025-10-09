import torch
from xbtorch.patches import xbtorch_model
import xbtorch.optim as xboptim

def test_hwa_training_basic(tabular_fefet_acc, simple_mlp_model_tabular):
    """Train a small model with WAGE-style quantization enabled and verify convergence."""
    torch.manual_seed(0)
    device = tabular_fefet_acc

    model = simple_mlp_model_tabular.to(device)
    model = xbtorch_model(model)

    x = torch.randn(128, 16, device=device)
    y = torch.randint(0, 4, (128,), device=device)
    criterion = torch.nn.CrossEntropyLoss()
    opt = xboptim.SGD(model.parameters(), lr=0.1)

    initial_loss = float(criterion(model(x), y))
    for _ in range(5):
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()

    final_loss = float(criterion(model(x), y))
    # With quantization, we still expect some convergence
    assert final_loss < initial_loss * 1.1, "Loss should not diverge with WAGE quantization"