import torch
import numpy as np

def test_committee(dataloader, all_models, device):
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = torch.flatten(inputs, start_dim=1)

            all_outputs = [] # Initialize an array to collect outputs from all models

            # Collect outputs from all models
            for model in all_models:
                outputs = model(inputs)
                all_outputs.append(outputs)

            # Stack and average outputs across all models
            all_outputs = torch.stack(all_outputs)
            averaged_outputs = torch.mean(all_outputs, dim=0)

            # Calculate predicted class
            _, predicted = torch.max(averaged_outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = 100 * correct / total
    # print(f'Committee Accuracy: {acc:.2f}%')
    return acc