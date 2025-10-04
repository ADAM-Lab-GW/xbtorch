XBTorch Documentation
=====================

XBTorch is a **PyTorch-native framework** for simulating *crossbar-based deep neural networks* 
with emerging memory technologies such as **ReRAM, FeFETs, PCM, and MTJs**.  

By providing a unified simulation interface, XBTorch enables researchers and engineers to:

- Model realistic **device-level behaviors** (variability, noise, nonlinearity),
- Perform **hardware-aware training** with quantization and gradient decomposition,
- Evaluate **fault-tolerant inference** on simulated crossbar arrays,
- Seamlessly integrate with existing PyTorch workflows.

Motivation
----------
Traditional deep learning relies on the von Neumann architecture, where the 
"memory wall" (data transfer between CPU/GPU and memory) becomes a bottleneck.  
In-memory computing with memristive devices bypasses this by performing 
multiply-accumulate (MAC) operations directly in memory arrays, potentially 
reducing energy consumption by orders of magnitude.

However, real devices are noisy, variable, and prone to faults. 
XBTorch bridges this gap by making **device-aware simulation** accessible 
directly from PyTorch, so that models can be trained, tested, and deployed 
with realistic hardware constraints in mind.

Key Features
------------
- **Device Models**: Analytical and tabular models for ReRAM, FeFET, and beyond.
- **Hardware-aware Training**: WAGE quantization, gradient decomposition, and noise injection.
- **Loss Landscape Visualization**: Tools to analyze robustness under parameter perturbations.
- **Hardware-aware Inference**: Stateful crossbar accelerators with encoding, mapping, and 
  fault-tolerance schemes (e.g., Layer Ensemble Averaging).
- **Seamless PyTorch Integration**: Minimal code changes required to port an existing model.

Getting Started
---------------
New to XBTorch? Start with these sections:

- :doc:`install` — set up your environment
- :doc:`quickstart` — run your first XBTorch-enabled model
- :doc:`concepts` — learn about device models, training, and inference

Links
-----
- GitHub: https://github.com/ADAM-Lab-GW/xbtorch
- Paper: [to be updated with arXiv/NeurIPS link]

XBTorch was introduced in:

.. note::

    O. Yousuf.
    *"XBTorch: A Framework for Optimization and Study of Emerging Neuromorphic Systems"*.  
    TBD. https://arxiv.org/abs/XYZ

.. toctree::
    :maxdepth: 2
    :caption: Get started
    :hidden:

    install
    quickstart
    concepts

.. toctree::
   :maxdepth: 4
   :caption: References
   :hidden:

   api
   citation

Reference
=========

:ref:`genindex` | :ref:`modindex`