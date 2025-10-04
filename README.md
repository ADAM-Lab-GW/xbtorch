# XBTorch

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](<ADD_DOCS_LINK>)
[![Arxiv](https://img.shields.io/badge/arXiv-XYZ-b31b1b.svg)](https://arxiv.org/abs/XYZ)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-compatible-orange)](https://pytorch.org)

---

**XBTorch** is a PyTorch-native framework for simulating **crossbar-based deep neural networks** with emerging memory technologies such as **ReRAM, FeFETs, PCM, and MTJs**.  

It enables researchers and engineers to:
- Model realistic **device-level behavior** (variability, noise, nonlinearity),
- Perform **hardware-aware training** with quantization and gradient decomposition,
- Evaluate **fault-tolerant inference** on simulated crossbar arrays,
- Seamlessly integrate with existing PyTorch models with minimal code changes.

👉 For detailed guides, please see the [XBTorch Documentation](<ADD_DOCS_LINK>).

---

## Dependencies & Installation

The recommended installation method is to create a lightweight virtual environment and install XBTorch in editable mode:

```bash
$ python -m venv .env
$ source .env/bin/activate
(.env) $ pip install -e xbtorch
````

This will install XBTorch in *editable mode*, allowing you to modify the source code directly.

For more detailed instructions (including optional dependencies and troubleshooting), review our [documentation](ADD_DOCS_LINK).

---

## Getting Started

Minimal code changes are needed to adapt PyTorch models for XBTorch:

```python
import xbtorch
import xbtorch.optim as xboptim
from xbtorch.patches import xbtorch_model
import torch.nn as nn

# Initialize
xbtorch.initialize()

# Define your model
model = MyMLP()
model = xbtorch_model(model)   # patch with XBTorch

# Optimizer
optimizer = xboptim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
```

For full examples (e.g., MNIST training, hardware-aware inference), see the `examples/` directory or the [documentation](ADD_DOCS_LINK).

---

## Citation

If you use this library, please cite this repository according to the information in `CITATION.cff` and/or the introductory paper:

```bibtex
@inproceedings{yousuf2025xbtorch,
  author    = {Yousuf, Osama and Glasmann, Andreu L. and Najmaei, Sina and Lueker-Boden, Martin and Adam, Gina C.},
  title     = {XBTorch: A Framework for Optimization and Study of Emerging Neuromorphic Systems},
  booktitle = {arXiv},
  year      = {2025},
  url       = {https://arxiv.org/abs/XYZ}
}
```

---

## Acknowledgements

This library was developed as a collaboration between:

* [The George Washington University (GWU)](https://gwu.edu)
* [DEVCOM Army Research Laboratory (ARL)]()
* [Western Digital Research](https://www.westerndigital.com/company/innovation/academic-collaborations)

The project is licensed under the **BSD-3 license** (see [LICENSE](LICENSE)).

---

## Contact and Collaboration

Research groups interested in collaborating are encouraged to reach out:

Osama Yousuf<br>
[Osama.Yousuf1@wdc.com](mailto:Osama.Yousuf1@wdc.com)<br>
R&D Engineering, Memory Technology<br>

Prof. Gina Adam<br>
[GinaAdam@gwu.edu](mailto:GinaAdam@gwu.edu)<br>
Adaptive Devices and Microsystems Group<br>
Department of Electrical and Computer Engineering<br>

Andreu L. Glasmann<br>
[Andreu.L.Glasmann.Civ@army.mil](mailto:andreu.l.glasmann.civ@army.mil)<br>
DEVCOM Army Research Lab<br>

---

## License

BSD-3 License. See the [LICENSE](LICENSE) file for details.
