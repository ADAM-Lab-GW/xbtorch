## xbtorch

xbtorch is a Python framework for simulating neural network models and investigating novel operations from the QSArray project. It is integrated with PyTorch for scalability, and supports wrapping over/binding to lower-level code in C++. Currently, we are working on supporting various device models for network training and inference operations, in order to investigate the role device non-idealities would play within the context of QSArray.

## Installation

#### Create and activate a virtual env

`python3 -m venv env`

`source env/bin/activate`

#### Install dependencies

`sudo apt-get install cmake, gcc-10, g++-10, python3-dev`

`sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 --slave /usr/bin/g++ g++ /usr/bin/g++-10 --slave /usr/bin/gcov gcov /usr/bin/gcov-10`

`pip3 install ./models/`

`pip3 install -r requirements.txt`

`git submodule update --init --recursive` 

#### Test the wrapper

`python3 tb_integration.py`

## Project Structure - TBD

`convergence_tests.py` - Testbench to verify whether or not the currently bound version of the QSArray converges.

`core.py` - Contains core methods for neural network testing, training, and dataset loading.

`device_models.py` - The most up-to-date entry point into xbtorch network training and testing. All simulation arguments can be set within this file.

`grid_optim.py` - Contains code for Optuna wrapping on top of xbtorch, can be used to start (and access saved) hyperparameter optimization studies saved on disk.

`main.py` - Old entry point into xbtorch. Does not contain device model integration.

`xbtorch.py` - Contains fundamentals of the xbtorch simulator (custom layers and operations)

`plots.py` - Contains various plotting methods to visualize common results.

`sw_mlp.py` - Completely detached SW version of an MLP - used primarily for testing.

`tb_bindings.py` - Testbench to verify the binding code works and is able to execute C++ methods from the golden model from within xbtorch.

`tb_fixedpoint.py` - Testbench for fixed point golden model - not up-to-date.

`nets/mlp.py` - Contains code for the underlying Two-layer MLP neural network. Also contains a One-layer MLP (not used) for testing purposes.

`models/` - Directory for pyBind11 bindings and configurations.

`models/src/qsarray` - Directory for the QSArray repository (will exist only if submodules were cloned)

`models/src/bindings.cpp`- Contains binding code for the golden model, which is how we integrate to the C++ QSArray codebase.

`device/jumptables` - Directory containing text files for various jump table models of devices.

`device/wage_initializer.py` - Contains initialization methods for WAGE quantization.

`device/wage_quantizer.py` - Contains core methods for quantization of weights, activations, gradients, and errors (WAGE).

`device/wage_util.py` - To contain helper methods required for WAGE quantization.

`device/device_util.py` - Contains core methods for device models - supports ideal, real, and stochastic (jump-table based) device models.

`device/quantization_cpu_np_infer.py` - Contains WAGE-based layers, not used within xbtorch. Instead, these are implemented as `NTLayerDev` within `xbtorch.py`

