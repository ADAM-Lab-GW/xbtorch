import torch.nn as nn
from ..patches import decorators

@decorators.xbtorch_layer
class Linear(nn.Linear): pass

@decorators.xbtorch_layer
class Conv2d(nn.Conv2d): pass

@decorators.xbtorch_layer
class RNN(nn.RNN): pass

@decorators.xbtorch_layer
class LSTM(nn.LSTM): pass

# Custom layer to select the last time step output in time-based models (RNNs, LSTMs)
class SelectLastStep(nn.Module):
    def forward(self, x):
        # x is a tuple (output, (h_n, c_n))
        lstm_out, _ = x
        # Extract the last time step
        return lstm_out[:, -1, :]