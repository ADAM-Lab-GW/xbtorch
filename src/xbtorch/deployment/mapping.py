import random
import torch
import numpy as np 
from collections import Counter

_MAX_ATTEMPTS = 1000

# Helper functions for mapping software weight matrices to the simulated array
def check_overlap(start1, shape1, start2, shape2):
    """
    Check if two submatrices overlap.
    
    Parameters:
    start1 (tuple): Starting index (row, col) of the first layer
    shape1 (tuple): Shape (rows, cols) of the first layer
    start2 (tuple): Starting index (row, col) of the second layer
    shape2 (tuple): Shape (rows, cols) of the second layer
    
    Returns:
    bool: True if the submatrices overlap, False otherwise
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

def probability_row_no_defects(n, m, r, defect_proportion):
    """
    Calculate the probability of finding a row of length `r` (in the same column) with no defects
    in a 2D grid of size `n x m` with a specified defect proportion.

    Parameters:
    - n: int, number of rows in the 2D grid
    - m: int, number of columns in the 2D grid
    - r: int, length of the row where no defects are desired
    - defect_proportion: float, proportion of cells in the grid that are defective (0 to 1)

    Returns:
    - float, probability of finding such a row

    >>> probability_row_no_defects(400, 400, 324, 0.05)
    2.4238338971471762e-05
    >>> probability_row_no_defects(400, 400, 324, 0.02)
    0.43725992923543056

    """
    p = defect_proportion

    # Probability that a specific cell has no defect
    p_no_defect = 1 - p

    # Probability that a specific column has a row of length `r` with no defects
    p_column_no_defect = p_no_defect ** r

    # Probability that no column has a row of length `r` with no defects
    p_no_column = (1 - p_column_no_defect) ** m

    # Probability of finding at least one column with a row of length `r` with no defects
    probability = 1 - p_no_column

    return probability

# XB Mapping functions - where should a particular weight matrix be mapped on the simulated chip?
def map_random(accelerator, layer_shape, beta=1, current_mappings=[]):
    """
    Returns a random index within the original tensor such that the layer fits inside of it
    and does not overlap with any already mapped submatrices.
    
    Parameters:
    chip_shape (tuple): Shape of the original tensor (e.g., (100, 10))
    layer_shape (tuple): Shape of the layer to fit (e.g., (5, 3))
    current_mappings (list): List of tuples with already mapped submatrices, where each tuple is
                               ((start_row, start_col), (layer_rows, layer_cols))
    
    Returns:
    tuple: Random starting index (row, col) within the original tensor
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

def map_layer_ensemble(accelerator, layer_shape, current_mappings=[], alpha=3, beta=1, debug=False):
    max_attempts = 10000
    window_size = 10
    chip = accelerator.chip
    chip_rows, chip_cols = chip.shape
    layer_rows, layer_cols = layer_shape
    defective_indices = accelerator.defect_map[0]
    
    # Ensure the layer can fit inside the tensor
    if layer_rows > chip_rows or layer_cols > chip_cols:
        raise ValueError("layer size is larger than the original tensor size.")
    # Calculate the maximum starting index to ensure the layer fits
    max_row_index = chip_rows - layer_rows
    max_col_index = chip_cols - layer_cols

    # Try to find a valid random starting index that does not overlap with existing submatrices
    mappings = [] # for each, we will compute \beta entries

    dims_remaining = np.ones(layer_rows) * beta
    zeros = np.zeros(layer_rows)

    attempts = max_attempts
    while not np.equal(dims_remaining, zeros).all() and attempts > 0 and len(mappings) < alpha:
        start_row = random.randint(0, max_row_index)
        start_col = random.randint(0, max_col_index)
        new_layer_start = (start_row, start_col)

        attempts -= 1

        # Check for overlap with already mapped submatrices
        overlap = False
        for mapped_start, mapped_shape in current_mappings:
            if check_overlap(new_layer_start, layer_shape, mapped_start, mapped_shape):
                overlap = True
                break
        
        if not overlap:
            # so we have a valid index
            # is it any useful?
            # usefulness means that it would improve the current solution state

            # count the number of stuck devices per output dimension
            submatrix = chip[start_row:start_row+layer_rows, start_col:start_col+layer_cols]

            # Filter indices that are within the submatrix boundaries and convert to relative indices
            relative_indices = []
            for index in range(len(defective_indices[0])):
                x, y = defective_indices[0][index], defective_indices[1][index]
                if (start_row <= x < start_row+layer_rows) and (start_col <= y < start_col+layer_cols):
                    relative_x = x - start_row
                    relative_y = y - start_col
                    relative_indices.append((relative_x, relative_y))

            # Count the occurrences per column in the submatrix
            column_counts = Counter(x for x, y in relative_indices)

            # Display the counts for each column in the submatrix
            column_counts_list = [column_counts.get(col, 0) for col in range(layer_rows)]

            # It would be useful if it adds to a defective dimensions
            # For very large layers, the probability of encountering a stuck device in a row is very high, which is why smaller layers are successfully getting mapped easily
            # Need to think of a new strategy to circumvent this
            # Perhaps a few options can be sampled, and the one that has the highest number of good devices, per row, can be used
            # This would allow us to use even rows with defective devices, but what about averaging during inference?
            # Luckily, I'm finding that larger layers are more generally tolerable of these defects - outputs are not easily faultered
            useful = False
            # can likely be vectorized
            for i in range(layer_rows):
                if (dims_remaining[i] > 0): # if there's a dim remaining
                    if (column_counts_list[i] == 0): # and if there's no defects for this
                        useful = True

            if not (useful):
                if (debug): print('Dimension not useful, skipping')
                continue
            else:
                mappings.append(new_layer_start)
                current_mappings.append((new_layer_start, layer_shape)) # very important to avoid conflicting mappings being produced
        
                # Update count of dims so far - if the layer has a good row, decrement the amount required
                for i in range(layer_rows):
                    if (column_counts_list[i] == 0): 
                        dims_remaining[i] = max(0, dims_remaining[i] - 1) 

    if (not np.equal(dims_remaining, zeros).all()):
        # raise ValueError("Could not find a valid non-overlapping layer position after max attempts.")
        print("Could not find a valid non-overlapping layer position after max attempts.")
        print(mappings, max_attempts)

    return mappings

def map_layer_ensemble_full_scan(accelerator, layer_shape, current_mappings=[], beta=1, debug=True):
    chip = accelerator.chip
    chip_rows, chip_cols = chip.shape
    layer_rows, layer_cols = layer_shape
    defective_indices = accelerator.defect_map[0]
    
    # Ensure the layer can fit inside the tensor
    if layer_rows > chip_rows or layer_cols > chip_cols:
        raise ValueError("layer size is larger than the original tensor size.")
    # Calculate the maximum starting index to ensure the layer fits
    max_row_index = chip_rows - layer_rows
    max_col_index = chip_cols - layer_cols

    stats = []
    offsets_so_far = []

    i = 0
    preload = False
    if (preload):
        stats = np.loadtxt(f'map_layer_ensemble_chip_{accelerator.name}.txt')
    else:
        for start_row in range(0, max_row_index):
            if (debug): print(f'Scanning row {start_row}/{chip.shape[0]}')
            for start_col in range(0, max_col_index):
                new_layer_start = (start_row, start_col)
                # Check for overlap with already mapped submatrices
                overlap = False
                for mapped_start, mapped_shape in current_mappings:
                    if check_overlap(new_layer_start, layer_shape, mapped_start, mapped_shape):
                        overlap = True
                        break
                
                if overlap: continue

                # count the number of stuck devices per output dimension
                submatrix = chip[start_row:start_row+layer_rows, start_col:start_col+layer_cols]

                # Filter indices that are within the submatrix boundaries and convert to relative indices
                relative_indices = []
                for index in range(len(defective_indices[0])):
                    x, y = defective_indices[0][index], defective_indices[1][index] 
                    if (start_row <= x < start_row+layer_rows) and (start_col <= y < start_col+layer_cols):
                        relative_x = x - start_row
                        relative_y = y - start_col
                        relative_indices.append((relative_x, relative_y))

                # Count the occurrences per column in the submatrix
                column_counts = Counter(x for x, y in relative_indices)

                # Display the counts for each column in the submatrix
                column_counts_list = [column_counts.get(col, 0) for col in range(layer_rows)]

                stuckCount = np.sum(column_counts_list)
                stats.append((stuckCount, start_row, start_col, layer_rows, layer_cols, column_counts_list))

        np.savetxt(f'map_layer_ensemble_chip_{accelerator.name}_layer_{layer_cols}_{layer_rows}.txt', stats)

    success = False
    stats = sorted(stats, key=lambda x: x[0], reverse=False) # sort by increasing # of stuck devices
    stuckCount, best_x, best_y, shape0, shape1, mapStuck = stats[0]
    stats = stats[1:]

    dims_remaining = np.ones(layer_cols) * beta
    zeros = np.zeros(layer_cols)

    # Add best found map in the given kernels to the offset list
    offsets_so_far.append([best_x, best_y, shape0, shape1, mapStuck])
    # Update count of dims so far - if the layer has a good row, decrement the amount required
    for i in range(layer_cols):
        if (mapStuck[i] == 0): dims_remaining[i] -= 1

    no_change = False

    dims_old = np.copy(dims_remaining)
    while not np.equal(dims_remaining, zeros).all() and not no_change:
        # Now in the remaining sorted offsets, we will continue to add until our list is exhausted
        for candidate_offset in stats:
            stuckCount, x, y, shape0, shape1, mapStuck = candidate_offset

            # does the current candidate offset overlap with some previously added offset?
            overlap = False
            for previous_offset in offsets_so_far:
                prev_x, prev_y, prev_shape0, prev_shape1, prev_mapStuck = previous_offset
                if check_overlap((x, y), (shape0, shape1), (prev_x, prev_y), (prev_shape0, prev_shape1)):
                    overlap = True
                    break
            
            if overlap: continue # if yes, discard and continue search
            else:
                # otherwise, valid candidate, but is it useful?
                if (debug): print('no collision', candidate_offset)
                if (debug): print(dims_remaining)
                # It would be useful if it adds to a defective dimensions
                useful = False
                for i in range(layer_cols):
                    if (dims_remaining[i] > 0): # if there's a dim remaining
                        if (mapStuck[i] == 0): # and if there's no defects for this
                            useful = True

                if not (useful):
                    if (debug): print('Dimension not useful, skipping')
                    continue

                # We have a dimension that is useful, add to our offset list
                offsets_so_far.append([x, y, shape0, shape1, mapStuck])
                # Update count of dims so far - if the layer has a good row, decrement the amount required
                for i in range(layer_cols):
                    if (mapStuck[i] == 0): 
                        dims_remaining[i] = max(0, dims_remaining[i] - 1)          

            if (np.equal(dims_remaining, zeros).all()):
                break # all conditions satisfied, break out

        if (np.equal(dims_old, dims_remaining).all()):
            if (debug): print('No Change!! Unable to satisfy conditions. Add more kernels for this layer.')
            no_change = True

        # for next iteration
        dims_old = np.copy(dims_remaining)

    if (np.equal(dims_remaining, zeros).all()):
        print("All conditions satisfied!!!")
        success = True

    if (debug):
        print(offsets_so_far)
    return offsets_so_far, success
