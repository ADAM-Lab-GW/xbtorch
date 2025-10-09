import torch.nn as nn

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