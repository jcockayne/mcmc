import numpy as np

def as_single_number(item):
    if type(item) is np.ndarray:
        return item.item()
    return item
