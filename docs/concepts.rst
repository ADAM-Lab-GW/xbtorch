Core Concepts
=============

XBTorch is organized around four pillars:

1. **Decomposition**
   - Defines how gradients are approximated or compressed
   - Examples: FullOuterProduct, NMF, SBPCA, SVD-based methods

2. **Deployment**
   - Maps software weights to hardware arrays
   - Encodings: simple binary, LEA1, LEA2, MAO
   - Mapping strategies: random placement, non-overlapping allocation
   - Fault-tolerant strategies: committee machines, collaborative classifiers

3. **Devices**
   - Models crossbar-level physics
   - Parameters: read/write noise, stuck devices, ADC/DAC quantization
   - Includes real prototypes like **Daffodil**

4. **Fault-Tolerance**
   - Improves robustness against hardware variability
   - Committee machine ensembling
   - Collaborative loss and logistic classifiers

.. figure:: _static/xbtorch-overview.png
   :align: center
   :alt: XBTorch workflow

   XBTorch workflow showing decomposition, deployment, and evaluation
   of neural networks on simulated accelerators.