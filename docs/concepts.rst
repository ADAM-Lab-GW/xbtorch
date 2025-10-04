Core Concepts
=============

XBTorch provides a modular framework to simulate memristive neural networks
with in-memory computing paradigms. Below we summarize the major components.

Device Models
-------------
Implemented in :mod:`xbtorch.devices`.

- **Analytical models**: Mathematical formulations with explicit noise terms
  (e.g., device-to-device variability, nonlinearity).
- **Tabular models**: Lookup-based representations of device switching,
  calibrated from experimental data.

Users can either:

- Use presets (ReRAM, FeFET models),
- Or create custom models via interpolation from measured datasets.

Hardware-aware Training
-----------------------
Implemented in :mod:`xbtorch.patches` and :mod:`xbtorch.quant`.

- All weight updates are inherently noisy.
- Supports **quantization-aware training** via WAGE (2-8-8-8 configurations for ternary weight networks).
- Users can plug in custom quantization functions for weights, activations, gradients, and errors.
- Compatible with QTorch for custom number formats.

Gradient Decomposition
----------------------
Implemented in :mod:`xbtorch.decomposition`.

- Large gradient matrices are approximated with **low-rank decompositions**.
- Supported methods:
  - Streaming Batch PCA (SBPCA)
  - Singular Value Decomposition (SVD)
  - Non-negative Matrix Factorization (NMF)
- Reduces communication cost in distributed training and extends device lifetime (fewer writes).

Loss Landscapes
---------------
Implemented in :mod:`xbtorch.loss`.

- Visualizes the optimization trajectory under device-level perturbations.
- Useful for studying robustness and parameter-space geometry of memristive networks.

Hardware-aware Inference
------------------------
Implemented in :mod:`xbtorch.deployment`.

- Stateful crossbar emulation with explicit mapping of weights to device arrays.
- Supports encoding schemes (e.g., differential weight encoding).
- Models system-level noise sources: ADC/DAC precision, stuck devices, write/read noise.
- Includes fault-tolerance algorithms:

  - Layer Ensemble Averaging (LEA),
  - Committee Machines,
  - Mapping with inner fault tolerance.

Design Philosophy
-----------------
- **PyTorch-native API**: Extend existing models with minimal modifications.
- **Technology-agnostic**: Works with ReRAM, FeFET, PCM, MTJ, or custom devices.
- **Research-oriented**: Modules for device modeling, training, and inference unify algorithmic exploration.

Next Steps
----------
- Explore provided tutorials in the ``examples/`` directory.
- Explore the :doc:`XBTorch API <api>`.
- Customize a device model using your own experimental data.