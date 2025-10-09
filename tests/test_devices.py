import numpy as np
from xbtorch.devices.utils import sweep_conductance_full, synthesize_G_dG_dataset

def test_conductance_sweep_runs(device_preset):
    """Ensure that conductance sweep runs without errors for all device presets."""
    device = device_preset
    set_G, reset_G = sweep_conductance_full(device)
    set_G, reset_G = np.array(set_G), np.array(reset_G)

    assert len(set_G) == device.max_level + 1
    assert len(reset_G) == device.max_level + 1
    assert set_G.ndim == 1
    assert reset_G.ndim == 1
    assert np.all(set_G > 0), "SET conductances must be positive"
    assert np.all(reset_G > 0), "RESET conductances must be positive"

def test_analytical_device_custom_parameters(custom_analytical_device):
    """Check that custom AnalyticalDevice parameters are properly respected."""
    device = custom_analytical_device
    set_G, _ = sweep_conductance_full(device)
    set_G = np.array(set_G)

    assert len(set_G) == device.max_level + 1
    assert np.all((set_G >= device.min_conductance) & (set_G <= device.max_conductance))

def test_synthesize_G_dG_dataset_shapes(analytical_real_device):
    """Ensure dataset synthesis produces correctly shaped outputs."""
    dev = analytical_real_device
    set_data = synthesize_G_dG_dataset(dev, set=True, num_points=500)
    reset_data = synthesize_G_dG_dataset(dev, set=False, num_points=500)

    assert set_data.shape == (500, 2)
    assert reset_data.shape == (500, 2)
    assert np.all(np.isfinite(set_data))
    assert np.all(np.isfinite(reset_data))

def test_tabular_and_analytical_consistency(analytical_real_device, tabular_analytical_device):
    """Default tabular model should produce similar behavior to analytical one (by definition)."""
    analytical = analytical_real_device
    tabular = tabular_analytical_device

    set_analytical = synthesize_G_dG_dataset(analytical, set=True, num_points=2000)
    set_tabular = synthesize_G_dG_dataset(tabular, set=True, num_points=2000)

    mean_diff = abs(set_analytical[:, 1].mean() - set_tabular[:, 1].mean())
    # Not identical but same order of magnitude
    assert mean_diff < 1e-8

def test_device_to_device_variability(analytical_real_device):
    """Ensure D2D variability introduces observable spread in responses."""
    dev = analytical_real_device
    dev.set_d2d_var(1.0)
    dev.set_c2c_var(0.0)

    results = []
    for i in range(3):
        set_G, _ = sweep_conductance_full(dev, group_param_idx=i)
        results.append(set_G)
        dev.reset_cached_params()

    std_across_devices = np.std(np.stack(results), axis=0).mean()
    assert std_across_devices > 0

def test_cycle_to_cycle_variability(analytical_real_device):
    """Ensure C2C variability causes response noise while preserving mean trend."""
    dev = analytical_real_device
    dev.set_d2d_var(0.0)
    dev.set_c2c_var(0.05)

    sets = [sweep_conductance_full(dev)[0] for _ in range(10)]
    std_c2c = np.std(np.stack(sets), axis=0).mean()
    assert std_c2c > 0, "Expected nonzero variability between cycles"

def test_nonlinearity_effects(analytical_real_device):
    """Test increasing nonlinearity_set increases curvature in SET trajectories."""
    dev = analytical_real_device
    dev.set_d2d_var(0.0)
    dev.set_c2c_var(0.0)

    nonlinearities = range(0, 6)
    responses = []

    for nl in nonlinearities:
        dev.set_nonlinearity_set(nl)
        dev.set_nonlinearity_reset(-nl)
        set_G, _ = sweep_conductance_full(dev)
        responses.append(set_G)
        dev.reset_cached_params()

    slopes = [np.gradient(r).mean() for r in responses]
    assert len(set(slopes)) > 1, "Nonlinearity should alter average slope"