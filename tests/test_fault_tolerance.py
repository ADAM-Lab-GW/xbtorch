import torch
import xbtorch
from xbtorch.deployment import encode_LEA1, SimpleFixedPoint, map_random
from xbtorch.patches import xbtorch_model
import torch.nn as nn
import pytest

from functools import partial

def test_stuck_faults_applied():
    device = torch.device("cpu")
    acc = SimpleFixedPoint(g_min=100, g_max=200, xb_size=(10, 10), stuck_percentage=0.2, stuck_mode="real", device=device)
    xbtorch.initialize(pytorch_device=device, inference_accelerator=acc)
    acc.initialize_chip()

    # Inspect the stored chip before readout. read_chip() clamps the -1
    # uninitialized sentinel to g_min, so it cannot distinguish unmapped cells.
    total_stuck = (acc._chip != -1).sum()
    expected_stuck = int(0.2 * acc._chip.numel())
    assert total_stuck == expected_stuck

@pytest.mark.parametrize("alpha_beta", [1, 3])
def test_fault_tolerance(mlp_model, alpha_beta):
    # Test successful mapping using LEA
    mapping_scheme = partial(map_random, beta=alpha_beta)

    device = torch.device("cpu")

    # TODO: use simple fixed point fixture with parametrization to support encode_LEA1 directly
    acc = SimpleFixedPoint(g_min=100, 
                           g_max=200, 
                           xb_size=(2500, 2500), 
                           stuck_percentage=0.2, 
                           stuck_mode="real", 
                           device=device,
                           weight_encoding_scheme=encode_LEA1,
                           xb_mapping_scheme=mapping_scheme)
    
    xbtorch.initialize(pytorch_device=device, 
                       inference_accelerator=acc)
    
    acc.initialize_chip()

    model = xbtorch_model(mlp_model)

    additional_args = {"alpha": alpha_beta, "beta": alpha_beta}

    model.initialize_array_mappings(output_polling_mode="avg", 
                                    additional_args=additional_args, 
                                    existing_mappings=[]
                                    )
    
    assert hasattr(model, "get_array_mappings")
    mappings = model.get_array_mappings()

    keys = ["Gneg", "Gpos"]
    for mapping in mappings:
        for key in keys:
            # ensure that mappings match expected redundancy count
            assert len(mapping[key]) == alpha_beta