import doctest

import datascience as ds
from datascience import util
import numpy as np
import pytest

def test_doctests():
    results = doctest.testmod(util, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert results.failed == 0

def test_make_array():
    test1 = ds.make_array(0)
    assert len(test1) == 1
    test2 = ds.make_array(2, 3, 4)
    assert sum(test2) == 9
    assert test2.dtype == "int64"
    test3 = ds.make_array("foo", "bar")
    assert test3.dtype == "<U3"


def test_percentile():
    assert ds.percentile(0, [1, 3, 5, 9]) == 1
    assert ds.percentile(25, [1, 3, 5, 9]) == 1
    assert ds.percentile(26, [1, 3, 5, 9]) == 3
    assert ds.percentile(55, [1, 3, 5, 9]) == 5
    assert ds.percentile(75, [1, 3, 5, 9]) == 5
    assert ds.percentile(76, [1, 3, 5, 9]) == 9

    f1 = ds.percentile(66)
    assert f1([1, 3, 5, 9]) == 5

    f2 = ds.percentile([65, 85])
    assert np.all(f2([9, 5, 3, 1]) == np.array([5, 9]))


def test_table_apply():
    data = np.ones([3, 100])
    data[1] = 2
    data[2] = 3
    # tab = ds.Table(data, ['a', 'b', 'c'])
    tab = ds.Table().with_columns('a', data[0], 'b', data[1], 'c', data[2])
    newtab = util.table_apply(tab, np.mean)
    assert newtab.num_rows == 1
    assert all(newtab['a'] == np.mean(tab['a']))

    newtab = util.table_apply(tab, lambda a: a+1)
    assert all(newtab['a'] == tab['a'] + 1)

    newtab = util.table_apply(tab, lambda a: a+1, subset=['b', 'c'])
    assert all(newtab['a'] == tab['a'])
    assert all(newtab['b'] == tab['b'] + 1)

    with pytest.raises(ValueError) as err:
        util.table_apply(tab, lambda a: a+1, subset=['b', 'd'])
        assert "Colum mismatch: ['d']" in str(err.value)


def _round_eq(a, b):
    if hasattr(a, '__len__'):
        return all(a == np.round(b))
    else:
        return (a == np.round(b)) == True


def test_minimize():
    assert _round_eq(2, ds.minimize(lambda x: (x-2)**2))
    assert _round_eq([2, 1], list(ds.minimize(lambda x, y: (x-2)**2 + (y-1)**2)))
    assert _round_eq(2, ds.minimize(lambda x: (x-2)**2, 1))
    assert _round_eq([2, 1], list(ds.minimize(lambda x, y: (x-2)**2 + (y-1)**2, [1, 1])))


def test_minimize_smooth():
    assert _round_eq(2, ds.minimize(lambda x: (x-2)**2, smooth=True))
    assert _round_eq([2, 1], list(ds.minimize(lambda x, y: (x-2)**2 + (y-1)**2, smooth=True)))
    assert _round_eq(2, ds.minimize(lambda x: (x-2)**2, 1, smooth=True))
    assert _round_eq([2, 1], list(ds.minimize(lambda x, y: (x-2)**2 + (y-1)**2, [1, 1], smooth=True)))


def test_minimize_array():
    assert _round_eq(2, ds.minimize(lambda x: (x[0]-2)**2, [0], array=True))
    assert _round_eq([2, 1], list(ds.minimize(lambda x: (x[0]-2)**2 + (x[1]-1)**2, [0, 0], array=True)))


def test_sample_proportions():
    uniform = ds.sample_proportions(1000, np.ones(50)/50)
    assert len(uniform) == 50 and _round_eq(1, sum(uniform))
    assert [x in (0, 0.5, 1) for x in ds.sample_proportions(2, ds.make_array(.2, .3, .5))]


def test_proportions_from_distribution():
    t = ds.Table().with_column('probs', np.ones(50)/50)
    u = ds.proportions_from_distribution(t, 'probs', 1000)
    assert t.num_columns == 1 and t.num_rows == 50
    assert u.num_columns == 2 and u.num_rows == 50
    uniform = u.column(1)
    assert len(uniform) == 50 and _round_eq(1, sum(uniform))
    assert [x in (0, 0.5, 1) for x in ds.sample_proportions(2, ds.make_array(.2, .3, .5))]
