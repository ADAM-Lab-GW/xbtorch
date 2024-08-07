# A Fault-Tolerant NNeural Network Architecture from "A Fault-Tolerant Neural Network Architecture" by Tao Liu et al
# Implementation based on: https://github.com/osama-usuf/A-Fault-Tolerant-Neural-Network-Architecture

import torch
import torch.nn as nn
import numpy as np

class CollaborativeLogisticClassifier(nn.Module):
    def __init__(self, input_size, num_classifiers):
        super(CollaborativeLogisticClassifier, self).__init__()
        self.classifiers = nn.Linear(input_size, num_classifiers, bias=False)
        self.significance = nn.Parameter(torch.ones(num_classifiers))

    def forward(self, x):
        return torch.sigmoid(self.classifiers(x))
    
# Custom loss function
class CollaborativeLoss(nn.Module):
    def __init__(self, codewords, model):
        super(CollaborativeLoss, self).__init__()
        self.codewords = codewords
        self.model = model
        self.reg_lambda = 0.1
        
    def forward(self, output, target, threshold=0.5):
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
    output_binary = (output > threshold).float()
    distances = torch.cdist(output_binary, codewords, p=0)
    return distances.argmin(dim=1)

def train_collaborative(model, loader, optimizer, loss_fn, codewords, device):
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
    '''
    Given a (patched) xbtorch model, replace the last layer with collaborative logistic classifiers in-place.
    '''
    # if doesn't have model attr, raise error
    if not hasattr(model, 'model') or len(model.model) < 2:
        raise ValueError("Module list not in expected format, likely a patching issue.")

    model.collaborative_classifier = CollaborativeLogisticClassifier(model.model[idx].weight.shape[1], num_classifiers)
    model.model[idx] = model.collaborative_classifier