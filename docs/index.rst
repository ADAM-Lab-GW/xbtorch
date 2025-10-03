.. XBTorch documentation master file, created by
   sphinx-quickstart on Tue Sep 30 22:03:23 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

XBTorch Documentation
=====================

**XBTorch** is a research framework for optimization and study of
memristive neural networks based on emerging neuromorphic systems. 
It provides a modular and extensible set of tools for modeling device non-idealities, 
mapping neural networks to crossbar accelerators, exploring fault-tolerant
architectures, etc. 

In particular, XBTorch streamlines:

- Hardware-aware decompositions and hardware-accurate training
- Weight encoding schemes with realistic device modeling
- Deployment strategies for inference across noisy and defective accelerators
- Fault-tolerant and analog noise-tolerant algorithm design

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