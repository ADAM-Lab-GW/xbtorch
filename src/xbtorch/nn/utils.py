import torch
from sklearn.metrics import confusion_matrix

def train_classifier(data_loader, model, criterion, optimizer, epoch, num_epochs=0, device="cpu", lr_decay_rate=1.0, log=False, fast=False):
    running_loss = 0.0
    epoch_loss = 0.0
    total_samples = 0
    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # inputs = torch.flatten(inputs, start_dim=1)
        inputs = inputs.view(1, -1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
        epoch_loss += loss.item()
        if log:  # Print every 100 mini-batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {running_loss/1:.4f}')
            running_loss = 0.0

        total_samples += inputs.size(0)

        if fast: break

    if (lr_decay_rate != 1.0):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay_rate

    return epoch_loss / total_samples

def test_classifier(data_loader, model, device, compute_cm=False, log=False):
    # TODO: This really shouldn't be test "classifier"
    # Evaluate the model
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    cm = None
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs = torch.flatten(inputs, start_dim=1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    if (log): print(f'Accuracy on test set: {acc:.2f}%')
    if (compute_cm): cm = confusion_matrix(all_labels, all_preds)
    return acc, cm

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