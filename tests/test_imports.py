import xbtorch

def test_imports():
    assert hasattr(xbtorch, "initialize"), "xbtorch.initialize not found"
    assert callable(xbtorch.initialize)

def test_initialization():
    xbtorch.initialize()
    assert xbtorch.get_xbtorch_param("initialized") is True