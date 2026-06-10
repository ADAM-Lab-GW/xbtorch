import torch
from xbtorch.deployment import SimpleFixedPoint
import xbtorch
from xbtorch.patches import xbtorch_model

def test_xb_eval_toggle(model_and_acc):
    model, _, _ = model_and_acc
    model.xb_eval(enable=True)
    assert model.xb_forward
    model.xb_eval(enable=False)
    assert not model.xb_forward

def test_initialize_array_mappings(model_and_acc):
    model, acc, _ = model_and_acc

    # Resize xb_size for this test
    acc.xb_size = (100, 100)
    model.initialize_array_mappings()
    mappings = model.get_array_mappings()

    assert isinstance(mappings, list)
    assert all(isinstance(layer, dict) for layer in mappings)

def test_plot_array_returns_tensor(model_and_acc):
    _, acc, _ = model_and_acc
    arr = acc.plot_array()
    assert isinstance(arr, torch.Tensor)

def test_stateless_accelerator(mlp_model_regular):
    device = "cpu"

    acc = SimpleFixedPoint(
        g_min=100,
        g_max=200,
        device=device,
        stateful=False
    )

    xbtorch.initialize(pytorch_device=device, inference_accelerator=acc)

    model = mlp_model_regular.to(device)
    model = xbtorch_model(model, replace_all=True)

    model.xb_eval(enable=False)

    return model, acc, device