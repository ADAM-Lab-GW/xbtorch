Quickstart
==========

This example reproduces Figure 1 from the paper:
mapping a simple PyTorch network onto a noisy accelerator.

.. code-block:: python

   import torch
   import torch.nn as nn
   from xbtorch.deployment import SimpleFixedPoint
   from xbtorch.decomposition import FullOuterProduct

   # Define a toy model
   model = nn.Sequential(
       nn.Linear(10, 5),
       nn.ReLU(),
       nn.Linear(5, 2)
   )

   # Instantiate an accelerator
   acc = SimpleFixedPoint(adc_bits=5, dac_bits=5, g_min=50, g_max=100)

   # Example input
   x = torch.randn(1, 10)

   # Forward pass
   y = model(x)

   print("Output:", y)

---

Next steps
-----------
- Learn about :doc:`concepts` in XBTorch
- See the :doc:`api` reference