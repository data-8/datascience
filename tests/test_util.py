import doctest

import datascience as ds
from datascience import util



def test_doctests():
    results = doctest.testmod(util)
    assert results.failed == 0


def test_percentile():
    assert ds.percentile(0, [1, 3, 5, 9]) == 1
    assert ds.percentile(33, [1, 3, 5, 9]) == 3
    assert ds.percentile(34, [1, 3, 5, 9]) == 5
    assert ds.percentile(66, [1, 3, 5, 9]) == 5
    assert ds.percentile(67, [1, 3, 5, 9]) == 9

    f1 = ds.percentile(67)
    assert f1([1, 3, 5, 9]) == 9

    f2 = ds.percentile(66)
    assert f2([1, 3, 5, 9]) == 5
