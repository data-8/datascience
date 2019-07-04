"""Utility functions"""

__all__ = ['make_array', 'percentile', 'plot_cdf_area', 'plot_normal_cdf',
           'table_apply', 'proportions_from_distribution',
           'sample_proportions', 'minimize']

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg', warn=False)
import matplotlib.pyplot as plt
from scipy import stats
from scipy import optimize
import functools
import math

# Change matplotlib formatting. TODO incorporate into a style?
plt.rcParams["patch.force_edgecolor"] = True

def make_array(*elements):
    """Returns an array containing all the arguments passed to this function.
    A simple way to make an array with a few elements.

    As with any array, all arguments should have the same type.

    >>> make_array(0)
    array([0])
    >>> make_array(2, 3, 4)
    array([2, 3, 4])
    >>> make_array("foo", "bar")
    array(['foo', 'bar'],
          dtype='<U3')
    >>> make_array()
    array([], dtype=float64)
    """
    if elements and all(isinstance(item, (int, np.integer)) for item in elements):
        # Specifically added for Windows machines where the default 
        # integer is int32 - see GH issue #339.
        return np.array(elements, dtype="int64")
    return np.array(elements)


def percentile(p, arr=None):
    """Returns the pth percentile of the input array (the value that is at
    least as great as p% of the values in the array).

    If arr is not provided, percentile returns itself curried with p

    >>> percentile(74.9, [1, 3, 5, 9])
    5
    >>> percentile(75, [1, 3, 5, 9])
    5
    >>> percentile(75.1, [1, 3, 5, 9])
    9
    >>> f = percentile(75)
    >>> f([1, 3, 5, 9])
    5
    """
    if arr is None:
        return lambda arr: percentile(p, arr)
    if hasattr(p, '__iter__'):
        return np.array([percentile(x, arr) for x in p])
    if p == 0:
        return min(arr)
    assert 0 < p <= 100, 'Percentile requires a percent'
    i = (p/100) * len(arr)
    return sorted(arr)[math.ceil(i) - 1]


def plot_normal_cdf(rbound=None, lbound=None, mean=0, sd=1):
    """Plots a normal curve with specified parameters and area below curve shaded
    between ``lbound`` and ``rbound``.

    Args:
        ``rbound`` (numeric): right boundary of shaded region

        ``lbound`` (numeric): left boundary of shaded region; by default is negative infinity

        ``mean`` (numeric): mean/expectation of normal distribution

        ``sd`` (numeric): standard deviation of normal distribution
    """
    shade = rbound is not None or lbound is not None
    shade_left = rbound is not None and lbound is not None
    inf = 3.5 * sd
    step = 0.1
    rlabel = rbound
    llabel = lbound
    if rbound is None:
        rbound = inf + mean
        rlabel = "$\infty$"
    if lbound is None:
        lbound = -inf + mean
        llabel = "-$\infty$"
    pdf_range = np.arange(-inf + mean, inf + mean, step)
    plt.plot(pdf_range, stats.norm.pdf(pdf_range, loc=mean, scale=sd), color='k', lw=1)
    cdf_range = np.arange(lbound, rbound + step, step)
    if shade:
        plt.fill_between(cdf_range, stats.norm.pdf(cdf_range, loc=mean, scale=sd), color='gold')
    if shade_left:
        cdf_range = np.arange(-inf+mean, lbound + step, step)
        plt.fill_between(cdf_range, stats.norm.pdf(cdf_range, loc=mean, scale=sd), color='darkblue')
    plt.ylim(0, stats.norm.pdf(0, loc=0, scale=sd) * 1.25)
    plt.xlabel('z')
    plt.ylabel('$\phi$(z)', rotation=90)
    plt.title("Normal Curve ~ ($\mu$ = {0}, $\sigma$ = {1}) "
              "{2} < z < {3}".format(mean, sd, llabel, rlabel), fontsize=16)
    plt.show()


# Old name
plot_cdf_area = plot_normal_cdf


def sample_proportions(sample_size, probabilities):
    """Return the proportion of random draws for each outcome in a distribution.

    This function is similar to np.random.multinomial, but returns proportions
    instead of counts.

    Args:
        ``sample_size``: The size of the sample to draw from the distribution.

        ``probabilities``: An array of probabilities that forms a distribution.

    Returns:
        An array with the same length as ``probability`` that sums to 1.
    """
    return np.random.multinomial(sample_size, probabilities) / sample_size


def proportions_from_distribution(table, label, sample_size,
                                  column_name='Random Sample'):
    """
    Adds a column named ``column_name`` containing the proportions of a random
    draw using the distribution in ``label``.

    This method uses ``np.random.multinomial`` to draw ``sample_size`` samples
    from the distribution in ``table.column(label)``, then divides by
    ``sample_size`` to create the resulting column of proportions.

    Args:
        ``table``: An instance of ``Table``.

        ``label``: Label of column in ``table``. This column must contain a
            distribution (the values must sum to 1).

        ``sample_size``: The size of the sample to draw from the distribution.

        ``column_name``: The name of the new column that contains the sampled
            proportions. Defaults to ``'Random Sample'``.

    Returns:
        A copy of ``table`` with a column ``column_name`` containing the
        sampled proportions. The proportions will sum to 1.

    Throws:
        ``ValueError``: If the ``label`` is not in the table, or if
            ``table.column(label)`` does not sum to 1.
    """
    proportions = sample_proportions(sample_size, table.column(label))
    return table.with_column('Random Sample', proportions)


def table_apply(table, func, subset=None):
    """Applies a function to each column and returns a Table.

    Args:
        ``table``: The table to apply your function to.

        ``func``: The function to apply to each column.

        ``subset``: A list of columns to apply the function to; if None,
            the function will be applied to all columns in table.

    Returns:
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


def minimize(f, start=None, smooth=False, log=None, array=False, **vargs):
    """Minimize a function f of one or more arguments.

    Args:
        f: A function that takes numbers and returns a number

        start: A starting value or list of starting values

        smooth: Whether to assume that f is smooth and use first-order info

        log: Logging function called on the result of optimization (e.g. print)

        vargs: Other named arguments passed to scipy.optimize.minimize

    Returns either:
        (a) the minimizing argument of a one-argument function
        (b) an array of minimizing arguments of a multi-argument function
    """
    if start is None:
        assert not array, "Please pass starting values explicitly when array=True"
        arg_count = f.__code__.co_argcount
        assert arg_count > 0, "Please pass starting values explicitly for variadic functions"
        start = [0] * arg_count
    if not hasattr(start, '__len__'):
        start = [start]

    if array:
        objective = f
    else:
        @functools.wraps(f)
        def objective(args):
            return f(*args)

    if not smooth and 'method' not in vargs:
        vargs['method'] = 'Powell'
    result = optimize.minimize(objective, start, **vargs)
    if log is not None:
        log(result)
    if len(start) == 1:
        return result.x.item(0)
    else:
        return result.x
