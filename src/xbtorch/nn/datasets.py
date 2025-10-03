"""
Utilities for preparing and loading character-level name datasets for
RNN classification tasks.

This module provides functions to read text files, convert Unicode names
to ASCII, encode names as tensors, and split datasets into PyTorch
DataLoaders. It also includes training and testing loops for RNN models.

Functions
---------
- :func:`findFiles` : List all files matching a path pattern.
- :func:`unicodeToAscii` : Convert Unicode string to ASCII given a vocabulary.
- :func:`read_lines` : Read lines from a file and convert to ASCII.
- :func:`name_to_tensor` : Encode a name as a tensor of one-hot vectors.
- :func:`get_dataloaders` : Split a dataset into train/test DataLoaders.
- :func:`train_rnn` : Train an RNN model on a given dataset.
- :func:`test_rnn` : Test an RNN model and return predictions and labels.
- :func:`get_names` : Fetch, preprocess, and return the names dataset.
- :class:`NamesDataset` : PyTorch Dataset class for names data.
"""

import subprocess
from zipfile import ZipFile
import os
from io import open
import glob
import unicodedata
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np

def findFiles(path):
    """
    Return a list of file paths matching a given pattern.

    Parameters
    ----------
    path : str
        Glob pattern for matching file paths.

    Returns
    -------
    list[str]
        List of file paths.
    """
    return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s, vocab):
    """
    Convert a Unicode string to plain ASCII characters, filtered by a vocabulary.

    Parameters
    ----------
    s : str
        Input Unicode string.
    vocab : str
        Allowed characters.

    Returns
    -------
    str
        ASCII-converted string containing only characters in vocab.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in vocab
    )

# Read a file and split into lines
def read_lines(filename, vocab):
    """
    Read lines from a text file and convert them to ASCII.

    Parameters
    ----------
    filename : str
        Path to the text file.
    vocab : str
        Allowed characters for conversion.

    Returns
    -------
    list[str]
        List of processed ASCII lines.
    """
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line, vocab) for line in lines]

# Turn a name into a <MAX_NAME_LENGTH x n_vocab>,
# i.e. an array of one-hot letter vectors
def name_to_tensor(MAX_NAME_LENGTH, line, vocab):
    """
    Convert a name into a tensor of one-hot letter vectors.

    Parameters
    ----------
    MAX_NAME_LENGTH : int
        Maximum length of a name (tensor rows).
    line : str
        Name to encode.
    vocab : str
        Character vocabulary.

    Returns
    -------
    torch.Tensor
        Tensor of shape (MAX_NAME_LENGTH, len(vocab)) with one-hot encoding.
    """
    tensor = torch.zeros(MAX_NAME_LENGTH, len(vocab))
    for li, letter in enumerate(line[:MAX_NAME_LENGTH]):
        tensor[li][vocab.find(letter)] = 1
    return tensor

def get_dataloaders(dataset, train_split=0.8, batch_size=32):
    """
    Split a dataset into training and testing DataLoaders.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset object to split.
    train_split : float, optional
        Fraction of data for training (default: 0.8).
    batch_size : int, optional
        Batch size for DataLoaders (default: 32).

    Returns
    -------
    tuple
        - train_loader : DataLoader
        - test_loader : DataLoader
    """
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_loader, test_loader

def train_rnn(model, data_loader, criterion, optimizer, epoch, num_epochs=10, device='cpu'):
    """
    Train an RNN model for one epoch on a given dataset.

    Parameters
    ----------
    model : torch.nn.Module
        RNN model.
    data_loader : DataLoader
        DataLoader for training data.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer.
    epoch : int
        Current epoch index.
    num_epochs : int, optional
        Total number of epochs (default: 10).
    device : str or torch.device, optional
        Device to train on ('cpu' or 'cuda', default: 'cpu').

    Returns
    -------
    float
        Average loss for the epoch.
    """
    # Move model to the specified device (GPU or CPU)
    model = model.to(device)
    
    # Set model to training mode
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Training loop
    """
    label dict {0: 7530, 2: 58, 4: 217, 9: 116, 17: 2938, 3: 575, 5: 575, 7: 242, 6: 185, 15: 1618, 16: 77, 14: 59, 8: 76, 12: 792, 10: 215, 11: 162, 13: 225, 1: 399}
    """
    for inputs, labels in data_loader:
        # Move inputs and labels to the device
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        # Zero the parameter gradients

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1) # grad can explode without this
        optimizer.step()
        
        # Track loss and accuracy
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Calculate average loss and accuracy for the epoch
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    return epoch_loss
    
# from sklearn.metrics import confusion_matrix
def test_rnn(model, data_loader,  device='cpu'):#, compute_cm=False):
    """
    Evaluate an RNN model on a dataset and return predictions and labels.

    Parameters
    ----------
    model : torch.nn.Module
        Trained RNN model.
    data_loader : DataLoader
        DataLoader for test data.
    device : str or torch.device, optional
        Device to perform evaluation (default: 'cpu').

    Returns
    -------
    tuple
        - all_preds : list[int]
            Predicted class indices.
        - all_labels : list[int]
            Ground truth class indices.
    """
    # Move model to the specified device (GPU or CPU)
    model = model.to(device)

    all_preds = []
    all_labels = []

    # Training loop
    with torch.no_grad():
        for inputs, labels in data_loader:

            # Move inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Concatenate all predictions and true labels
    # all_preds = torch.cat(all_preds, dim=0)
    # all_labels = torch.cat(all_labels, dim=0)

    return all_preds, all_labels

# Define PyTorch Dataset
class NamesDataset(Dataset):
    """
    PyTorch Dataset for character-level name data.

    Attributes
    ----------
    X : list[torch.Tensor]
        Input tensors representing names.
    y : list[torch.Tensor]
        Label tensors representing class indices.
    data_dict : dict
        Original mapping of categories to names.
    classes : list[str]
        Sorted list of all categories.
    n_classes : int
        Number of categories.
    """
    def __init__(self, data_dict, classes, X, y):
        self.X = X
        self.y = y
        self.data_dict = data_dict
        self.classes = sorted(classes)
        self.n_classes = len(classes)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_names(vocab, train_split, batch_size, MAX_NAME_LENGTH=8, extract_to="./"):
    """
    Fetch, preprocess, and return the names dataset.

    Downloads from PyTorch tutorial dataset if not already extracted,
    converts names to ASCII tensors, and returns train/test DataLoaders.

    Parameters
    ----------
    vocab : str
        Allowed character vocabulary.
    train_split : float
        Fraction of data for training.
    batch_size : int
        Batch size for DataLoaders.
    MAX_NAME_LENGTH : int, optional
        Maximum name length (default: 8).
    extract_to : str, optional
        Directory to extract dataset (default: './').

    Returns
    -------
    tuple
        - dataset : NamesDataset
        - train_loader : DataLoader
        - test_loader : DataLoader
    """
    # Check if the data is already extracted
    if os.path.exists(f"{extract_to}/data/names"):
        print(f"Data already exists, skipping download and extraction.")
    else:
        # Download the dataset
        filename = "_xbtemp_data.zip"
        url = "https://download.pytorch.org/tutorial/data.zip"
        download_path = f"{url}/{filename}"
        result = subprocess.run(["wget", "-O", filename, url], check=True)
        # Extract
        if download_path.endswith(".zip"):
            with ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(extract_to)
                print(f"File extracted to: {extract_to}")

        # Remove temp archive file
        if os.path.exists(filename):
            os.remove(filename)

    # Pre-processing
    # Character-level vocab representing all letters
    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []

    X, y = [], []
    for filename in findFiles('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename, vocab)
        category_lines[category] = lines

        for line in lines:
            X.append(name_to_tensor(MAX_NAME_LENGTH, line, vocab))
            y.append(torch.tensor(all_categories.index(category), dtype=torch.long))

    # Convert to tensors
    dataset = NamesDataset(category_lines, all_categories, X, y)
    train_loader, test_loader = get_dataloaders(dataset, train_split=train_split, batch_size=batch_size)

    return dataset, train_loader, test_loader