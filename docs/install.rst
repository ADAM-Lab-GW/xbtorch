Installation
============

XBTorch requires Python 3.10+ and integrates seamlessly with PyTorch. 
It is distributed via GitHub as well as PyPI and can be installed in a few simple steps.

.. note::

   We recommend working in a virtual environment.

Installing from source (Recommended)
------------------------------------
Clone the repository and install using pip:

.. code-block:: bash

   git clone https://github.com/ADAM-Lab-GW/xbtorch.git
   cd xbtorch
   pip install -e .

This will install XBTorch in editable mode so you can directly modify the source code. 

Installing from PyPI
---------------------
Simply install via pip. 

.. code-block:: bash

   pip install xbtorch

Testing your installation
-------------------------
After installation, test that the library imports correctly:

.. code-block:: python

   import xbtorch
   xbtorch.initialize()

If no errors are raised, you are ready to use XBTorch.