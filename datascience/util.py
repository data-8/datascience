"""Utility functions"""

__all__ = ['percentile', 'plot_cdf_area', 'table_apply']


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg', warn=False)
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

def plot_cdf_area(rbound, lbound=None, mean=0, sd=1):
    """Plots a normal curve with specified parameters and area below curve shaded
    between ``lbound`` and ``rbound``.

    Args:
        ``rbound`` (numeric): right boundary of shaded region

        ``lbound`` (numeric): left boundary of shaded region; by default is negative infinity

        ``mean`` (numeric): mean/expectation of normal distribution

        ``sd`` (numeric): standard deviation of normal distribution
    """
    INF = 3.5 * sd
    llabel = lbound
    if lbound is None:
        lbound = -INF + mean
        llabel = "-$\infty$"
    pdf_range = np.arange(-INF + mean, INF + mean, 0.1)
    plt.plot(pdf_range, stats.norm.pdf(pdf_range, loc=mean, scale=sd), color='k', lw=1)
    cdf_range = np.arange(lbound, rbound + .1, 0.1)
    plt.fill_between(cdf_range, stats.norm.pdf(cdf_range, loc=mean, scale=sd), color='gold')
    plt.ylim(0, stats.norm.pdf(0, loc=0, scale=sd) * 1.25)
    plt.xlabel('z')
    plt.ylabel('$\phi$(z)', rotation=0)
    plt.title("Normal Curve ~ ($\mu$ = {0}, $\sigma$ = {1}) "
              "{2} < z < {3}".format(mean, sd, llabel, rbound), fontsize=16)
    plt.show()


def table_apply(table, func, subset=None):
    """Applies a function to each column and returns a Table.

    Uses pandas `apply` under the hood, then converts back to a Table

    Parameters
    ----------
    table : instance of Table
        The table to apply your function to
    func : function
        Any function that will work with DataFrame.apply
    subset : list | None
        A list of columns to apply the function to. If None, function
        will be applied to all columns in table

    Returns
    -------
    tab : instance of Table
        A table with the given function applied. It will either be the
        shape == shape(table), or shape (1, table.shape[1])
    """
    from . import Table
    df = table.to_df()

    if subset is not None:
        # Iterate through columns
        subset = np.atleast_1d(subset)
        if any([i not in df.columns for i in subset]):
            err = np.where([i not in df.columns for i in subset])[0]
            err = "Column mismatch: {0}".format(
                [subset[i] for i in err])
            raise ValueError(err)
        for col in subset:
            df[col] = df[col].apply(func)
    else:
        df = df.apply(func)
    if isinstance(df, pd.Series):
        # Reshape it so that we can easily convert back
        df = pd.DataFrame(df).T
    tab = Table.from_df(df)
    return tab
