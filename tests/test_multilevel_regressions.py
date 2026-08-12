import pytest
import torch
import torch.nn as nn

import xbtorch
from xbtorch.deployment import SimpleFixedPoint
from xbtorch.patches import xbtorch_model


class TwoSameShapeLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4, bias=False)
        self.fc2 = nn.Linear(4, 4, bias=False)

    def forward(self, inputs):
        return self.fc2(self.fc1(inputs))


class NegativeDominantLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1, bias=False)

    def forward(self, inputs):
        return self.fc(inputs)


def make_stateless_accelerator(**overrides):
    options = {
        "g_min": 50.0,
        "g_max": 100.0,
        "v_read": 0.3,
        "read_noise": 0.0,
        "write_noise": 0.0,
        "stateful": False,
        "adc_bits": 16,
        "dac_bits": 16,
        "programming_seed": 17,
        "weight_encoding_args": {"bits": 8, "zero_tol": 0.0},
    }
    options.update(overrides)
    return SimpleFixedPoint(**options)


def patch_stateless(model, accelerator):
    xbtorch.initialize(
        pytorch_device="cpu",
        inference_accelerator=accelerator,
    )
    model = xbtorch_model(model, replace_all=True)
    model.xb_eval(enable=True)
    return model


def patched_layers(model):
    return [
        module
        for module in model.modules()
        if hasattr(module, "_xb_programming_seed")
    ]


def programmed_conductances(accelerator, layer):
    return accelerator.map_weights_to_array_stateless(
        layer.weight.detach(),
        programming_seed=layer._xb_programming_seed,
    )


def conductances_are_equal(left, right):
    return all(
        torch.equal(left_array, right_array)
        for left_polarity, right_polarity in zip(left, right)
        for left_array, right_array in zip(left_polarity, right_polarity)
    )


def test_stateless_programming_error_is_static_across_repeated_mapping():
    accelerator = make_stateless_accelerator(write_noise=4.0)
    weights = torch.tensor(
        [[-1.0, -0.4, 0.0, 0.3], [0.8, -0.2, 0.5, -0.7]]
    )

    first = accelerator.map_weights_to_array_stateless(
        weights,
        programming_seed=123,
    )
    second = accelerator.map_weights_to_array_stateless(
        weights,
        programming_seed=123,
    )

    assert conductances_are_equal(first, second)


def test_programming_regeneration_does_not_advance_dynamic_rng():
    accelerator = make_stateless_accelerator(write_noise=4.0)
    weights = torch.tensor(
        [[-1.0, -0.4, 0.0, 0.3], [0.8, -0.2, 0.5, -0.7]]
    )

    torch.manual_seed(2468)
    expected = torch.rand(8)

    torch.manual_seed(2468)
    accelerator.map_weights_to_array_stateless(
        weights,
        programming_seed=123,
    )
    observed = torch.rand(8)

    assert torch.equal(observed, expected)


def test_only_write_noise_gives_identical_repeated_forwards():
    accelerator = make_stateless_accelerator(write_noise=4.0)
    # Isolate programming/read behavior from the separately tested converters.
    accelerator.DAC_quantize = lambda values: values
    accelerator.ADC_quantize = lambda values: values

    model = patch_stateless(TwoSameShapeLayers(), accelerator)
    inputs = torch.tensor([[0.2, -0.4, 0.6, 0.8]])

    first = model(inputs)
    second = model(inputs)

    assert torch.equal(first, second)


def test_read_noise_changes_repeated_forwards():
    accelerator = make_stateless_accelerator(read_noise=5.0)
    # Make read noise the only stochastic component in this test.
    accelerator.DAC_quantize = lambda values: values
    accelerator.ADC_quantize = lambda values: values

    model = patch_stateless(TwoSameShapeLayers(), accelerator)
    inputs = torch.tensor([[0.2, -0.4, 0.6, 0.8]])

    first = model(inputs)
    second = model(inputs)

    assert not torch.equal(first, second)


def test_same_shaped_layers_have_independent_programming_fields():
    accelerator = make_stateless_accelerator(
        write_noise=4.0,
        programming_seed=90,
    )
    model = TwoSameShapeLayers()
    shared_weights = torch.linspace(-0.9, 0.9, 16).reshape(4, 4)
    with torch.no_grad():
        model.fc1.weight.copy_(shared_weights)
        model.fc2.weight.copy_(shared_weights)

    model = patch_stateless(model, accelerator)
    first_layer, second_layer = patched_layers(model)

    assert first_layer._xb_programming_seed == 90
    assert second_layer._xb_programming_seed == 91
    assert not conductances_are_equal(
        programmed_conductances(accelerator, first_layer),
        programmed_conductances(accelerator, second_layer),
    )


def test_reprogramming_is_reproducible_and_changes_hardware_instance():
    accelerator = make_stateless_accelerator(write_noise=4.0)
    model = patch_stateless(TwoSameShapeLayers(), accelerator)

    def snapshot():
        return [
            programmed_conductances(accelerator, layer)
            for layer in patched_layers(model)
        ]

    model.reprogram_xb(123)
    first = snapshot()
    model.reprogram_xb(124)
    different = snapshot()
    model.reprogram_xb(123)
    repeated = snapshot()

    assert accelerator.programming_seed == 123
    assert all(
        conductances_are_equal(left, right)
        for left, right in zip(first, repeated)
    )
    assert any(
        not conductances_are_equal(left, right)
        for left, right in zip(first, different)
    )


def test_multilevel_bits_reach_stateless_encoder():
    weights = torch.tensor([[-1.0, -0.4, 0.0, 0.4, 1.0]])
    two_bit = make_stateless_accelerator(
        weight_encoding_args={"bits": 2, "zero_tol": 0.0}
    )
    eight_bit = make_stateless_accelerator(
        weight_encoding_args={"bits": 8, "zero_tol": 0.0}
    )

    two_pos, two_neg = two_bit.map_weights_to_array_stateless(
        weights,
        programming_seed=1,
    )
    eight_pos, eight_neg = eight_bit.map_weights_to_array_stateless(
        weights,
        programming_seed=1,
    )
    two_reconstructed = (two_pos[0] - two_neg[0]) / (
        two_bit.g_max - two_bit.g_min
    )
    eight_reconstructed = (eight_pos[0] - eight_neg[0]) / (
        eight_bit.g_max - eight_bit.g_min
    )

    torch.testing.assert_close(
        two_reconstructed,
        torch.tensor([[-1.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0]]),
    )
    torch.testing.assert_close(eight_reconstructed, weights)
    assert not torch.equal(two_reconstructed, eight_reconstructed)


def test_multilevel_bits_reach_stateful_encoder():
    accelerator = SimpleFixedPoint(
        g_min=50.0,
        g_max=100.0,
        read_noise=0.0,
        write_noise=0.0,
        stateful=True,
        xb_size=(4, 6),
        weight_encoding_args={"bits": 2, "zero_tol": 0.0},
    )
    weights = torch.tensor([[-1.0, -0.4, 0.0, 0.4, 1.0]])

    accelerator.map_weights_to_array(
        weights,
        pos_idxs=[(0, 0)],
        neg_idxs=[(1, 0)],
    )
    reconstructed = (
        accelerator._chip[0:1, 0:5] - accelerator._chip[1:2, 0:5]
    ) / (accelerator.g_max - accelerator.g_min)

    torch.testing.assert_close(
        reconstructed,
        torch.tensor([[-1.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0]]),
    )


def test_negative_dominant_weights_use_maximum_absolute_scale():
    accelerator = make_stateless_accelerator(
        weight_encoding_args={"bits": 2, "zero_tol": 0.0}
    )
    model = NegativeDominantLayer()
    with torch.no_grad():
        model.fc.weight.copy_(torch.tensor([[-2.0, 1.0]]))

    model = patch_stateless(model, accelerator)
    output = model(torch.ones(1, 2))

    # Two-bit magnitude encoding maps 1/2 to 2/3. Scaling by gamma=2
    # therefore realizes [-2, 4/3], whose dot product with [1, 1] is -2/3.
    torch.testing.assert_close(
        output,
        torch.tensor([[-2.0 / 3.0]]),
        rtol=0.0,
        atol=1e-3,
    )


def test_zero_dac_and_adc_inputs_remain_finite_and_exactly_zero():
    accelerator = make_stateless_accelerator(adc_bits=8, dac_bits=8)
    zeros = torch.zeros(3, 5)

    dac_output = accelerator.DAC_quantize(zeros)
    adc_output = accelerator.ADC_quantize(zeros)

    assert torch.isfinite(dac_output).all()
    assert torch.isfinite(adc_output).all()
    assert torch.equal(dac_output, zeros)
    assert torch.equal(adc_output, zeros)


@pytest.mark.parametrize("converter_name", ["DAC_quantize", "ADC_quantize"])
def test_fixed_point_converter_is_deterministic(converter_name):
    accelerator = make_stateless_accelerator(adc_bits=4, dac_bits=4)
    values = torch.tensor([[1.0, 0.12345, -0.45678]])
    converter = getattr(accelerator, converter_name)

    outputs = [converter(values) for _ in range(32)]

    assert all(torch.equal(outputs[0], output) for output in outputs[1:])



@pytest.mark.parametrize("converter_name", ["DAC_quantize", "ADC_quantize"])
@pytest.mark.parametrize("shape", [(2, 4), (2, 3, 4)])
def test_converter_scale_is_independent_per_vmm_vector(
    converter_name,
    shape,
):
    accelerator = make_stateless_accelerator(adc_bits=4, dac_bits=4)
    converter = getattr(accelerator, converter_name)

    values = torch.tensor([0.50, 0.19, -0.31, 0.07])
    baseline_input = values.expand(shape).clone()
    perturbed_input = baseline_input.clone()

    # Change only the last VMM vector's full scale. With a tensor-global
    # maximum this also changes unrelated batch items/tokens.
    perturbed_input.reshape(-1, shape[-1])[-1] *= 100.0

    baseline = converter(baseline_input)
    perturbed = converter(perturbed_input)

    torch.testing.assert_close(
        perturbed.reshape(-1, shape[-1])[:-1],
        baseline.reshape(-1, shape[-1])[:-1],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("converter_name", ["DAC_quantize", "ADC_quantize"])
def test_converter_preserves_shape_for_single_vector(converter_name):
    accelerator = make_stateless_accelerator(adc_bits=4, dac_bits=4)
    values = torch.tensor([0.50, 0.19, -0.31, 0.07])

    assert getattr(accelerator, converter_name)(values).shape == values.shape