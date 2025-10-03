"""
Crossbar mapping utilities for weight placement.

This module provides helper functions for mapping software weight
matrices onto physical crossbar subarrays in simulation. Proper mapping
ensures that each weight matrix fits within the hardware dimensions and
does not overlap with previously placed matrices.

Functions
---------
- :func:`check_overlap` : Check whether two mapped submatrices overlap.
- :func:`map_random` : Randomly assign non-overlapping positions for
  weight matrices on the simulated chip.

Notes
-----
Mapping is a critical part of hardware-aware simulation, as overlapping
placements would result in invalid or conflicting conductance states.
"""

import random
import torch
import numpy as np 
from collections import Counter

_MAX_ATTEMPTS = 1000

# Helper functions for mapping software weight matrices to the simulated array
def check_overlap(start1, shape1, start2, shape2):
    """
    Check if two submatrices overlap in the crossbar array.

    Parameters
    ----------
    start1 : tuple[int, int]
        Starting index (row, col) of the first submatrix.
    shape1 : tuple[int, int]
        Shape (rows, cols) of the first submatrix.
    start2 : tuple[int, int]
        Starting index (row, col) of the second submatrix.
    shape2 : tuple[int, int]
        Shape (rows, cols) of the second submatrix.

    Returns
    -------
    bool
        True if the submatrices overlap, False otherwise.

    Examples
    --------
    >>> check_overlap((0, 0), (2, 2), (1, 1), (2, 2))
    True
    >>> check_overlap((0, 0), (2, 2), (3, 3), (2, 2))
    False
    """
    r1, c1 = start1
    r2, c2 = start2
    rows1, cols1 = shape1
    rows2, cols2 = shape2

    # Check if one rectangle is to the left of the other
    if c1 + cols1 <= c2 or c2 + cols2 <= c1:
        return False
    
    # Check if one rectangle is above the other
    if r1 + rows1 <= r2 or r2 + rows2 <= r1:
        return False
    
    return True

# XB Mapping functions - where should a particular weight matrix be mapped on the simulated chip?
def map_random(accelerator, layer_shape, beta=1, current_mappings=[]):
    """
    Randomly map a weight matrix to non-overlapping locations on a crossbar array.

    Each weight matrix is mapped onto the simulated accelerator's chip
    at random valid positions. Multiple redundant mappings can be
    requested (via ``beta``) for ensemble or fault-tolerant encoding.

    Parameters
    ----------
    accelerator : GenericAccelerator
        The accelerator instance containing chip dimensions
        (``rows``, ``columns``).
    layer_shape : tuple[int, int]
        Shape (rows, cols) of the layer to map.
    beta : int, optional
        Number of redundant mappings to generate (default: 1).
    current_mappings : list[tuple[tuple[int, int], tuple[int, int]]], optional
        Existing mappings to ensure no overlaps are introduced.
        Each entry is ``((start_row, start_col), (rows, cols))``.

    Returns
    -------
    list[tuple[int, int]]
        List of starting indices (row, col) for each mapping.

    Raises
    ------
    ValueError
        If the layer does not fit in the array dimensions or if no valid
        non-overlapping placement is found after the maximum number of attempts.

    Notes
    -----
    - The number of attempts is capped by ``_MAX_ATTEMPTS`` (default: 1000).
    - Ensures that newly placed layers do not overlap with existing
      mapped regions.
    """
    chip_rows, chip_cols = accelerator.rows, accelerator.columns
    layer_rows, layer_cols = layer_shape
    
    # Ensure the layer can fit inside the tensor
    if layer_rows > chip_rows or layer_cols > chip_cols:
        raise ValueError("layer size is larger than the original tensor size.")
    
    # Calculate the maximum starting index to ensure the layer fits
    max_row_index = chip_rows - layer_rows
    max_col_index = chip_cols - layer_cols
    
    # Try to find a valid random starting index that does not overlap with existing submatrices
    mappings = [] # for each, we will compute \beta entries
    for j in range(beta):
        attempts = _MAX_ATTEMPTS  # Limit the number of attempts to avoid infinite loops
        while attempts > 0:
            start_row = random.randint(0, max_row_index)
            start_col = random.randint(0, max_col_index)
            new_layer_start = (start_row, start_col)
            
            # Check for overlap with already mapped submatrices
            overlap = False
            for mapped_start, mapped_shape in current_mappings:
                if check_overlap(new_layer_start, layer_shape, mapped_start, mapped_shape):
                    overlap = True
                    break
            
            if not overlap:
                mappings.append(new_layer_start)
                current_mappings.append((new_layer_start, layer_shape)) # very important to avoid conflicting mappings being produced
                break
            
            attempts -= 1

    if (len(mappings) != beta):
        raise ValueError("Could not find a valid non-overlapping layer position after max attempts.")

    return mappings