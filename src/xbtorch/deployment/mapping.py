import random
import torch

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

def get_random_index(chip_shape, layer_shape, current_mappings=[]):
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
    chip_rows, chip_cols = chip_shape
    layer_rows, layer_cols = layer_shape
    
    # Ensure the layer can fit inside the tensor
    if layer_rows > chip_rows or layer_cols > chip_cols:
        raise ValueError("layer size is larger than the original tensor size.")
    
    # Calculate the maximum starting index to ensure the layer fits
    max_row_index = chip_rows - layer_rows
    max_col_index = chip_cols - layer_cols
    
    # Try to find a valid random starting index that does not overlap with existing submatrices
    idxs = []
    for i in range(2): # one for Gpos, another for Gneg
        attempts = 10000  # Limit the number of attempts to avoid infinite loops
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
                idxs.append([new_layer_start])
                current_mappings.append((new_layer_start, layer_shape)) # very important to avoid conflicting mappings being produced
                break
            
            attempts -= 1
    
    if (len(idxs) != 2):
        raise ValueError("Could not find a valid non-overlapping layer position after max attempts.")
    return idxs