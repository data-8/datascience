"""Utility functions"""

import numpy as np

def percentile(p, arr=None):
    """Returns the pth percentile of the input array (the value that is at
    least as great as p% of the values in the array)

    If arr is not provided, percentile returns itself curried with p

    >>> percentile(67, [1, 3, 5, 9])
    9
    >>> percentile(66, [1, 3, 5, 9])
    5
    >>> f = percentile(66)
    >>> f([1, 3, 5, 9])
    5
    """
    if arr is None:
        return lambda arr: percentile(p, arr)
    return np.percentile(arr, p, interpolation='higher')
