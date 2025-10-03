"""
Fault-tolerant neural network architectures for crossbar-based accelerators.

This module implements the **Collaborative Logistic Classifier (CLC)**
method from Liu et al., *"A Fault-Tolerant Neural Network Architecture"*.
CLC introduces redundancy into the final classification layer using
multiple logistic sub-classifiers and error-correcting codewords. This
design improves robustness to hardware defects, device variability, and
noise commonly encountered in memristive crossbars.

The workflow includes:

- Collaborative classifiers (:class:`CollaborativeLogisticClassifier`)
- Custom loss function balancing BCE and Hamming distance penalties
  (:class:`CollaborativeLoss`)
- Codeword construction with large Hamming distances
  (:func:`dnn_favorable_searching_code`)
- Decoding methods to map predictions back to class indices
  (:func:`variable_length_decode`)
- Training and testing utilities
  (:func:`train_collaborative`, :func:`test_collaborative`)
- Model patching helper to replace standard classifiers
  (:func:`add_collaborative_logistic_classifiers`)

References
----------
- Tao Liu et al., "A Fault-Tolerant Neural Network Architecture".
- Original implementation:
  https://github.com/osama-usuf/A-Fault-Tolerant-Neural-Network-Architecture
"""

import torch
import torch.nn as nn
import numpy as np

class CollaborativeLogisticClassifier(nn.Module):
    """
    Collaborative logistic classifier layer.

    This layer replaces the final classification layer in a network
    with multiple logistic classifiers, one per "committee member".
    Each classifier outputs a probability (via sigmoid), and the final
    prediction is decoded using collaborative error-correcting codewords.

    Parameters
    ----------
    input_size : int
        Dimension of the input feature vector.
    num_classifiers : int
        Number of collaborative logistic classifiers.

    Attributes
    ----------
    classifiers : nn.Linear
        Linear layer with `num_classifiers` outputs, no bias.
    significance : nn.Parameter
        Trainable parameter vector controlling the relative importance
        of each classifier.
    """

    def __init__(self, input_size, num_classifiers):
        super(CollaborativeLogisticClassifier, self).__init__()
        self.classifiers = nn.Linear(input_size, num_classifiers, bias=False)
        self.significance = nn.Parameter(torch.ones(num_classifiers))

    def forward(self, x):
        """
        Forward pass through the collaborative classifiers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, input_size)``.

        Returns
        -------
        torch.Tensor
            Sigmoid probabilities of shape ``(batch_size, num_classifiers)``.
        """
        return torch.sigmoid(self.classifiers(x))
    
class CollaborativeLoss(nn.Module):
    """
    Custom loss for collaborative logistic classifiers.

    This combines binary cross-entropy (BCE) loss with a penalty term
    based on the Hamming distance between the predicted and target
    codewords. The penalty encourages outputs to stay closer to the
    target codeword in code space, increasing fault tolerance.

    Parameters
    ----------
    codewords : torch.Tensor
        Matrix of target codewords, shape ``(num_classes, num_classifiers)``.
    model : nn.Module
        Collaborative model using :class:`CollaborativeLogisticClassifier`.

    Attributes
    ----------
    reg_lambda : float
        Regularization weight for the Hamming distance penalty (default=0.1).
    """

    def __init__(self, codewords, model):
        super(CollaborativeLoss, self).__init__()
        self.codewords = codewords
        self.model = model
        self.reg_lambda = 0.1
        
    def forward(self, output, target, threshold=0.5):
        """
        Compute collaborative loss.

        Parameters
        ----------
        output : torch.Tensor
            Model predictions, shape ``(batch_size, num_classifiers)``.
        target : torch.Tensor
            True class indices, shape ``(batch_size,)``.
        threshold : float, optional
            Threshold for binarizing predictions. Default: 0.5.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        bce_loss = nn.BCELoss(reduction='none')
        losses = bce_loss(output, self.codewords[target])
        
        # Calculate significance based on Hamming distance between predicted codeword and target codeword
        with torch.no_grad():
            pred = (output > threshold).float()
            hamming_distances = torch.cdist(pred, self.codewords[target], p=0)
            sigma = hamming_distances.min(dim=1)[0]

        weighted_losses = (1-self.reg_lambda)*losses + self.reg_lambda * sigma.unsqueeze(1) #* self.model.collaborative_classifier.significance.unsqueeze(0)

        return weighted_losses.mean() # average over the mini-batch dimension

def dnn_favorable_searching_code(conf_matrix, num_classifiers, hamming_distance=3):
    """
    Construct favorable error-correcting codewords for collaborative classifiers.

    Based on a confusion matrix, assigns binary codewords to classes
    such that inter-class Hamming distances are maximized, ensuring
    robustness against errors in individual classifiers.

    Parameters
    ----------
    conf_matrix : array-like
        Confusion matrix from training or validation, shape
        ``(num_classes, num_classes)``.
    num_classifiers : int
        Number of collaborative classifiers (codeword length).
    hamming_distance : int, optional (default=3)
        Minimum required Hamming distance between codewords.

    Returns
    -------
    tuple
        - dict : mapping of class index → codeword (as integer).
        - torch.Tensor : codeword matrix of shape
          ``(num_classes, num_classifiers)``, with binary entries.

    Notes
    -----
    - If no codewords are found at the desired Hamming distance, the
      distance is iteratively reduced.
    """
    # Part 1: Prepare the searching table (Lines 1-8)
    def hamm_dist(code1, code2):
        return sum(c1 != c2 for c1, c2 in zip(code1, code2))
    
    num_classes = conf_matrix.shape[0]
    code_length = num_classifiers
    T = set()
    while len(T) == 0 and hamming_distance >= 3:
        for i in range(1, 2**num_classifiers):
            binary_code = format(i, f'0{code_length}b')
            if all(hamm_dist(binary_code, format(t, f'0{code_length}b')) >= hamming_distance for t in T):
                T.add(i)
        if len(T) == 0:
            hamming_distance -= 1
    
    # Part 2: Searching code
    O = set()
    x = 1
    while len(O) < num_classes and x < 2**num_classifiers:
        binary_code = format(x, f'0{code_length}b')
        if all(hamm_dist(binary_code, format(o, f'0{code_length}b')) >= hamming_distance for o in O):
            O.add(x)
        x += 1

    # Part 3: Code-tp-class assignment
    O = list(O)
    codeword_dict = {}
    C = np.array(conf_matrix)
    k = num_classes
    while len(C) > 0 and k > 0:
        max_index = np.argmax(C)
        i, j = divmod(max_index, num_classes)
        if i != j and i not in codeword_dict:
            codeword_dict[i] = O.pop()
            k -= 1
        C[i, j] = -1
        
    codeword_matrix = torch.tensor([list(map(int, f'{x:0{code_length}b}')) for k, x in codeword_dict.items()], dtype=torch.float32)
    return codeword_dict, codeword_matrix

def variable_length_decode(output, codewords, threshold=0.5):
    """
    Decode model outputs to class predictions using codewords.

    Parameters
    ----------
    output : torch.Tensor
        Model outputs, shape ``(batch_size, num_classifiers)``.
    codewords : torch.Tensor
        Codeword matrix of shape ``(num_classes, num_classifiers)``.
    threshold : float, optional (default=0.5)
        Threshold for binarizing outputs.

    Returns
    -------
    torch.Tensor
        Predicted class indices, shape ``(batch_size,)``.
    """
    output_binary = (output > threshold).float()
    distances = torch.cdist(output_binary, codewords, p=0)
    return distances.argmin(dim=1)

def train_collaborative(model, loader, optimizer, loss_fn, codewords, device):
    """
    Train a collaborative classifier model.

    Parameters
    ----------
    model : nn.Module
        Collaborative model (with :class:`CollaborativeLogisticClassifier`).
    loader : torch.utils.data.DataLoader
        DataLoader providing training batches.
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    loss_fn : nn.Module
        Loss function, typically :class:`CollaborativeLoss`.
    codewords : torch.Tensor
        Codeword matrix, shape ``(num_classes, num_classifiers)``.
    device : str or torch.device
        Device for computation.
    """
    model.train()
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = torch.flatten(inputs, start_dim=1)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

def test_collaborative(model, loader, codewords, device):
    """
    Evaluate a collaborative classifier model.

    Parameters
    ----------
    model : nn.Module
        Collaborative model.
    loader : torch.utils.data.DataLoader
        DataLoader providing evaluation batches.
    codewords : torch.Tensor
        Codeword matrix, shape ``(num_classes, num_classifiers)``.
    device : str or torch.device
        Device for computation.

    Returns
    -------
    float
        Accuracy over the dataset (0–1).
    """
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = torch.flatten(inputs, start_dim=1)
            output = model(inputs)
            pred = variable_length_decode(output, codewords)
            correct += pred.eq(labels).sum().item()
    return correct / len(loader.dataset)

def add_collaborative_logistic_classifiers(model, num_classifiers, idx=-2):
    """
    Replace the last layer of a model with collaborative classifiers.

    Parameters
    ----------
    model : nn.Module
        Model (patched xbtorch model) containing a ``model`` attribute
        with a module list.
    num_classifiers : int
        Number of collaborative logistic classifiers to insert.
    idx : int, optional (default=-2)
        Index of the layer to replace.

    Raises
    ------
    ValueError
        If the model does not contain a ``model`` attribute in the
        expected format.
    """
    # if doesn't have model attr, raise error
    if not hasattr(model, 'model') or len(model.model) < 2:
        raise ValueError("Module list not in expected format, likely a patching issue.")

    model.collaborative_classifier = CollaborativeLogisticClassifier(model.model[idx].weight.shape[1], num_classifiers)
    model.model[idx] = model.collaborative_classifier