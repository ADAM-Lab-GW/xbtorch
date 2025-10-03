"""
Committee machine (ensemble) evaluation module.

Provides functions to evaluate ensembles of models by aggregating
their predictions. The main function `test_committee` computes
classification accuracy by averaging outputs from all models in the
committee.

This approach improves robustness and generalization, particularly
for networks deployed on noisy or non-ideal hardware.
"""

import torch
import numpy as np

def test_committee(dataloader, all_models, device):
    """
    Evaluate a committee (ensemble) of models on a dataset.

    This method implements the *committee machines* approach, where
    predictions from multiple models are aggregated by averaging their
    outputs before classification. It is commonly used to improve
    robustness and generalization, particularly in the presence of
    hardware noise or device variability.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader providing input samples and labels for evaluation.
    all_models : list of torch.nn.Module
        A list of models to be ensembled. Each model must be compatible
        with the given input shape and return class logits.
    device : str or torch.device
        Device on which evaluation will be performed (e.g., "cpu" or "cuda").

    Returns
    -------
    float
        Classification accuracy of the committee (in percentage).

    Notes
    -----
    - Each model in the committee processes the same input batch.
    - Predictions are averaged across models before applying
      :func:`torch.max` to determine the predicted class.
    - Returns only the accuracy; predicted labels and ground truth
      are accumulated internally but not returned.

    References
    ----------
    - Joksas, "Committee machines—a universal method to deal with non-idealities 
      in memristor-based neural networks", Nature Communications, 2020.
    - Opitz & Maclin, "Popular Ensemble Methods: An Empirical Study",
      Journal of Artificial Intelligence Research, 1999.
    """
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
    return acc