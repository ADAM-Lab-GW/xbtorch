import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
class YinYangMLP(nn.Module):
    def __init__(self):
        super(YinYangMLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(4, 12),
            nn.Tanh(),
            nn.Linear(12, 6),
            nn.Tanh(),
            nn.Linear(6, 3)
        )

    def forward(self, x):
        x = self.model(x)
        return x