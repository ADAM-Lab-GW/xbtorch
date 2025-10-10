import torch
from xbtorch.deployment import SimpleFixedPoint
import xbtorch
from xbtorch.patches import xbtorch_model

# def test_xb_eval_toggle(model_and_acc):
#     model, _, _ = model_and_acc
#     model.xb_eval(enable=True)
#     assert model.xb_forward
#     model.xb_eval(enable=False)
#     assert not model.xb_forward

# def test_initialize_array_mappings(model_and_acc):
#     model, acc, _ = model_and_acc

#     # Resize xb_size for this test
#     acc.xb_size = (100, 100)
#     model.initialize_array_mappings()
#     mappings = model.get_array_mappings()

#     assert isinstance(mappings, list)
#     assert all(isinstance(layer, dict) for layer in mappings)

# def test_plot_array_returns_tensor(model_and_acc):
#     _, acc, _ = model_and_acc
#     arr = acc.plot_array()
#     assert isinstance(arr, torch.Tensor)

def test_stateless_accelerator(mlp_model_regular):
    device = "cpu"
    acc = SimpleFixedPoint(
        g_min=100,
        g_max=200,
        xb_size=(1000, 1000),
        device=device,
        stateful=False
    )

    xbtorch.initialize(pytorch_device=device, inference_accelerator=acc)

    model = mlp_model_regular.to(device)
    model = xbtorch_model(model)

    print(model)
    # ONGOING : Patching is passing but not working (see model.py)
    # need to make sure layers are still replaced even if stateless
    # XBPatching for module ('fc1', Linear(in_features=784, out_features=150, bias=False)) is not defined, using as is.
    # XBPatching for module ('fc2', Linear(in_features=150, out_features=10, bias=False)) is not defined, using as is.
    # XBPatching for module ('a1', ReLU()) is not defined, using as is.
    # SimpleMLPNoModel(
    # (fc1): Linear(in_features=784, out_features=150, bias=False)
    # (fc2): Linear(in_features=150, out_features=10, bias=False)
    # (a1): ReLU()
    # )
    # .

    return model, acc, device