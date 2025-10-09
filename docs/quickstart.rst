Quickstart
==========

This guide walks you through adapting an existing PyTorch script into an XBTorch-enabled simulation.

Minimal Example
---------------
Suppose you have a simple MLP in PyTorch:

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim

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

   model = SimpleMLP(500, 100, 10)
   optimizer = optim.SGD(model.parameters(), lr=0.01)
   criterion = nn.CrossEntropyLoss()

With XBTorch, only minimal changes are needed:

.. code-block:: python

   import torch.nn as nn
   import xbtorch
   import xbtorch.optim as xboptim
   from xbtorch.patches import xbtorch_model

   # ... SimpleMLP definition skipped for brevity

   # Initialize XBTorch with default settings
   xbtorch.initialize()

   # Define model
   model = SimpleMLP(500, 100, 10)

   # Patch model for crossbar simulation
   model = xbtorch_model(model)

   # Define hardware-aware optimizer
   optimizer = xboptim.SGD(model.parameters(), lr=0.01)
   criterion = nn.CrossEntropyLoss()

   # Training loop
   for epoch in range(10):
      model.train()
      ...
      # Standard PyTorch training logic applies

Key Idea
--------
The transition from PyTorch to XBTorch is seamless:

- Replace ``torch.optim`` with :mod:`xbtorch.optim`.
- Wrap your model with :mod:`xbtorch.patches.model.xbtorch_model()`.
- Initialize the framework with :mod:`xbtorch.initialize()`.

From here, you can activate specific modules for:

- Device modeling (realistic FeFET/ReRAM devices),
- Hardware-aware training (noisy weight updates, quantization),
- Hardware-aware inference (fault-tolerant deployment).

Next Steps
----------
We recommend reviewing :doc:`Core Concepts <concepts>` next.