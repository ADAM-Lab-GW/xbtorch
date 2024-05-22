import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import xbtorch
import xbtorch.nn as xbnn
import xbtorch.optim as xboptim

from xbtorch.decomposition import FullSVD, TruncatedSVD, NMF, SBPCA, FullOuterProduct
from xbtorch.devices import AnalyticalIdeal, AnalyticalReal, TabularAnalyticalReal

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = xbnn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = xbnn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
if __name__ == '__main__':

    # Check if CUDA is available and select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # device = 'cpu'
    torch.set_default_device(device)

    # Decomposition Algorithms
    decomposition_algorithm = FullSVD()
    decomposition_algorithm = TruncatedSVD(rank=2)
    decomposition_algorithm = NMF(rank=2, streaming=False)
    decomposition_algorithm = NMF(rank=2, streaming=True)
    decomposition_algorithm = SBPCA(rank=1)
    decomposition_algorithm = FullOuterProduct()
    decomposition_algorithm = None

    # Device Types
    xb_device = AnalyticalIdeal()
    xb_device = AnalyticalReal()
    xb_device = TabularAnalyticalReal()
    # xb_device = None

    # todo: weight_range should be selectable from here
    # todo: rename device_type to device
    xbtorch.initialize(decomposition_algorithm=decomposition_algorithm, 
                       device_type=xb_device,
                       pytorch_device=device
                       )

    # Define transforms to apply to the data
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize the image data
    ])

    # Load the MNIST training and test datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True, generator=torch.Generator(device=device))
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, generator=torch.Generator(device=device))

    # Define the model
    input_size = 784  # 28x28 image flattened to a vector
    hidden_size = 64
    output_size = 10  # 10 classes (digits 0-9)
    model = SimpleMLP(input_size, hidden_size, output_size).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    optimizer = xboptim.SGD(model.parameters(), lr=0.1)

    # Train the model
    import time

    num_epochs = 1
    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            start = time.time()
            inputs = inputs.view(-1, 784)  # Flatten the input images
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            if i % 1 == 0:  # Print every 100 mini-batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
            end = time.time()
            print('iter time', end - start)
    print('Finished Training')

    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, 784)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100 * correct / total:.2f}%')