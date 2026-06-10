import xbtorch
from xbtorch.patches import xbtorch_model

def test_imports():
    assert hasattr(xbtorch, "initialize"), "xbtorch.initialize not found"
    assert callable(xbtorch.initialize)


def test_initialization(xbtorch_hwa_initialized):
    """Test that xbtorch initializes correctly with HWA fixture."""
    assert xbtorch.get_xbtorch_param("initialized") is True

def test_device_initialization(simple_fixedpoint_accelerator):
    """Test SimpleFixedPoint device properties."""
    acc = simple_fixedpoint_accelerator
    assert acc.g_min == 133
    assert acc.g_max == 233
    assert acc.get_xb_size() == (2500, 2500)

def test_model_patching(mlp_model, simple_fixedpoint_accelerator):
    """Test that xbtorch_model patches models correctly."""

    # Stateful
    
    # Model without HWA enabled
    model = mlp_model
    xbtorch.initialize()  # baseline initialization
    patched_model = xbtorch_model(model)
    # xb_eval should NOT exist when no accelerator passed to xbtorch.initialize
    assert not hasattr(patched_model, "xb_eval")

    # Model with HWA enabled
    xbtorch.initialize(inference_accelerator=simple_fixedpoint_accelerator)
    patched_model = xbtorch_model(model)
    # xb_eval should exist
    assert hasattr(patched_model, "xb_eval")
