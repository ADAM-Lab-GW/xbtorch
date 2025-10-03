"""
Deployment (mapping, encoding, etc.) of solutions to inference accelerators  
"""
from .base import Daffodil, SimpleFixedPoint
from .mapping import map_random
from .encoding import encode_simple_binary, encode_MAO, encode_LEA1, encode_LEA2
from .metrics import compute_error
from .correction import train_collaborative, test_collaborative, CollaborativeLoss, add_collaborative_logistic_classifiers, dnn_favorable_searching_code
from .committee import test_committee