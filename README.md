# XBTorch

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://musical-doodle-pg28p2m.pages.github.io/)
[![Arxiv](https://img.shields.io/badge/arXiv-XYZ-b31b1b.svg)](https://arxiv.org/abs/2601.07086)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-compatible-orange)](https://pytorch.org)

---

**XBTorch** is a PyTorch-native framework for simulating **crossbar-based deep neural networks** with emerging memory technologies such as **ReRAM, FeFETs, PCM, and MTJs**.  

It enables researchers and engineers to:
- Model realistic **device-level behavior** (variability, noise, nonlinearity),
- Perform **hardware-aware training** with quantization and gradient decomposition,
- Evaluate **fault-tolerant inference** on simulated crossbar arrays,
- Seamlessly integrate with existing PyTorch models with minimal code changes.

👉 For detailed guides, please see the [XBTorch Documentation](https://musical-doodle-pg28p2m.pages.github.io/).

---

## Dependencies & Installation

The recommended installation method is to create a lightweight virtual environment and install XBTorch in editable mode:

```bash
$ python -m venv .env
$ source .env/bin/activate
(.env) $ pip install -e xbtorch
````

This will install XBTorch in *editable mode*, allowing you to modify the source code directly.

For more detailed instructions (including optional dependencies and troubleshooting), review our [documentation](https://musical-doodle-pg28p2m.pages.github.io/).

---

## Getting Started

Minimal code changes are needed to adapt PyTorch models for XBTorch:

```python
import xbtorch
import xbtorch.optim as xboptim
from xbtorch.patches import xbtorch_model
import torch.nn as nn

# Define a simple 2-layer perceptron network
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size, bias=False),
        )

    def forward(self, x):
        x = x.view(-1, self.input_size)  # Flatten the image
        x = self.model(x)
        return x

# Initialize
xbtorch.initialize()

# Define your model
model = SimpleMLP(10, 5, 2)
model = xbtorch_model(model)   # patch with XBTorch

# Optimizer
optimizer = xboptim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# ... Implement your training loop as usual!
```

For full examples (e.g., hardware-aware training and inference, fault-tolerance, etc.), see the `examples/` directory or the [documentation](https://musical-doodle-pg28p2m.pages.github.io/).

---

## Citation

If you use this library, please cite this repository according to the information in `CITATION.cff` and/or the introductory paper:

```bibtex
@inproceedings{yousuf2025xbtorch,
  author    = {Yousuf, Osama and Glasmann, Andreu L. and Lueker-Boden, Martin and Najmaei, Sina and Adam, Gina C.},
  title     = {XBTorch: A Unified Framework for Modeling and Co-Design of Crossbar-Based Deep Learning Accelerators},
  booktitle = {arXiv},
  year      = {2026},
  url       = {https://arxiv.org/abs/2601.07086}
}
```

---

## Acknowledgements

This library was developed as a collaboration between:

* [The George Washington University (GWU)](https://gwu.edu)
* [DEVCOM Army Research Laboratory (ARL)](https://www.orau.org/ARLFellowship/fellowship-types/how-to-apply-journeyman.htm)
* [Western Digital Research](https://www.westerndigital.com/company/innovation/academic-collaborations)

The project is licensed under the **BSD-3 license** (see [LICENSE](LICENSE)).

---

## Contact and Collaboration

Research groups interested in collaborating are encouraged to reach out:

Osama Yousuf<br>
[Osama.Yousuf1@wdc.com](mailto:Osama.Yousuf1@wdc.com)<br>
Western Digital Research<br>
R&D Engineering, Memory Technology<br>

Prof. Gina Adam<br>
[GinaAdam@gwu.edu](mailto:GinaAdam@gwu.edu)<br>
Adaptive Devices and Microsystems Group<br>
Department of Electrical and Computer Engineering<br>
George Washington University<br>

Andreu L. Glasmann<br>
[Andreu.L.Glasmann.Civ@army.mil](mailto:andreu.l.glasmann.civ@army.mil)<br>
DEVCOM Army Research Lab<br>

---

## License


BSD-3 License. See the [LICENSE](LICENSE) file for details.

