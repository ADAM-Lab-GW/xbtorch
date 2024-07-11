import numpy as np
import random
import torch


def find_nearest(array, values):
    # Expand dimensions for broadcasting
    array = array.unsqueeze(0)  
    values = values.unsqueeze(1) 
    abs_diff = torch.abs(array - values)
    indices = abs_diff.argmin(dim=1)
    return indices

def find_nearest_2d(array, values):
    # Expand dimensions for broadcasting
    if (array.dim() == 1): array = array.unsqueeze(0)

    array = array.unsqueeze(1)  
    values = torch.Tensor(values).unsqueeze(1).unsqueeze(2) 

    abs_diff = torch.abs(array - values)  
    indices = abs_diff.argmin(dim=2).squeeze(1)

    return indices

def stochastic_round(x):
    a = torch.floor(x)
    return a + ((x - a) > torch.rand_like(x)).float()

def sweep_conductance_full(device, group_param_idx=(0, 0)):
    set_Gs = [device.min_conductance_set]
    G = set_Gs[0]
    for i in range(device.max_level):
        G = device.write(torch.Tensor([G]), numPulse=torch.Tensor([1]), group_param_idx=group_param_idx).item()
        set_Gs.append(G)

    reset_Gs = [device.max_conductance_reset]
    G = reset_Gs[0]
    for i in range(device.max_level):
        G = device.write(torch.Tensor([G]), numPulse=torch.Tensor([-1]), group_param_idx=group_param_idx).item()
        reset_Gs.append(G)

    return set_Gs, reset_Gs

def synthesize_G_dG_dataset(device, set=True, num_points=1000, min_delta_ratio=1/1000, disable_filtering=False):
    '''
    Analogous to Algorithm 2 from:
    Device Modeling Bias in ReRAM-based Neural Network Simulations
    https://ieeexplore.ieee.org/abstract/document/10024104

    Instead of using known mean and standard deviation profiles, just use the device write operation 
    '''

    dataset = np.zeros((num_points, 2)) # generate (G, dG) dataset with total points specified by `num_points`
    numPulse = 1 if set else -1
    for point in range(num_points):
        valid = False
        while (not valid):
            G = random.uniform(device.min_conductance, device.max_conductance) # generate a random conductance value from possible device range
            newG = device.write(torch.Tensor([G]), numPulse=torch.Tensor([numPulse])).item()
            deltaG = newG - G
            # additional constraints can be added here for experimental realism
            if (disable_filtering): valid = True
            elif (newG >= device.min_conductance and newG <= device.max_conductance and abs(deltaG/G) > min_delta_ratio):  valid = True

            dataset[point] = [G, deltaG]

    return dataset

def print_num_unique_values(tensor):
    unique_values = torch.unique(tensor)
    num_unique_values = unique_values.numel()  # numel() returns the number of elements in the tensor
    print(f'Number of unique values: {num_unique_values}', torch.max(tensor), torch.min(tensor))
    # print(tensor)

    # import matplotlib.pyplot as plt
    
    # # Extract the weights from the layer
    # weights = tensor.numpy().flatten()
    
    # # Plot the histogram of the weights
    # plt.figure(figsize=(10, 6))
    # plt.hist(weights, bins=30, edgecolor='black')
    # plt.title('Histogram of Linear Layer Weights')
    # plt.xlabel('Weight values')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.show()


def train_classifier(dataloader, model, criterion, optimizer, epoch, num_epochs, device):
    running_loss = 0.0
    epoch_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = torch.flatten(inputs, start_dim=1)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
        epoch_loss += loss.item()
        if i % 1 == 0:  # Print every 100 mini-batches
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss/1:.4f}')
            running_loss = 0.0

    return running_loss

def test_classifier(dataloader, model, device):
    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = torch.flatten(inputs, start_dim=1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f'Accuracy on test set: {acc:.2f}%')
    return acc