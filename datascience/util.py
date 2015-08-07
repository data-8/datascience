"""Utility functions"""

import numpy as np

def percentile(arr, p):
    """Returns the pth percentile of the input array (the value that is at
    least as great as p% of the values in the array)

    >>> percentile([1, 3, 5, 9], 67)
    9
    >>> percentile([1, 3, 5, 9], 66)
    5
    """
    return np.percentile(arr, p, interpolation='higher')
