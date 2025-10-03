"""
Gradient decomposition algorithms 
"""

from .svd import FullSVD, TruncatedSVD
from .nmf import NMF
from .sbpca import SBPCA
from .base import FullOuterProduct