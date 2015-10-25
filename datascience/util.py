"""Utility functions"""

__all__ = ['percentile', 'plotCDF']


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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

def plotCDFarea(rbound, lbound=None, mean=0, sd=1):
    """Plots a normal curve with specified parameters and area below curve shaded
    between ``lbound`` and ``rbound``.

    Args:
        ``rbound`` (numeric): right boundary of shaded region

        ``lbound`` (numeric): left boundary of shaded region; by default is negative infinity

        ``mean`` (numeric): mean/expectation of normal distribution

        ``sd`` (numeric): standard deviation of normal distribution
    """
    llabel = lbound
    if lbound is None:
        lbound = sd * -3.5 + mean
        llabel = "-$\infty$"
    z = np.arange(sd * -3.5 + mean, sd * 3.5 + mean, 0.1)
    plt.plot(z, stats.norm.pdf(z, loc=mean, scale=sd), color='k', lw=1)
    w = np.arange(lbound, rbound + .1, 0.1)
    plt.fill_between(w, stats.norm.pdf(w, loc=mean, scale=sd), color='gold')
    plt.ylim(0, stats.norm.pdf(0, loc=0, scale=sd) * 1.25)
    plt.xlabel('z')
    plt.ylabel('$\phi$(z)', rotation=0)
    plt.title("Normal Curve ~ ($\mu$ = {0}, $\sigma$ = {1}) {2} < z < {3}".format(mean, sd, llabel, rbound), fontsize=16)
    plt.show()
