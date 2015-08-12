import pytest

from datascience import *

def test_percentile():
    assert percentile(0, [1, 3, 5, 9]) == 1
    assert percentile(33, [1, 3, 5, 9]) == 3
    assert percentile(34, [1, 3, 5, 9]) == 5
    assert percentile(66, [1, 3, 5, 9]) == 5
    assert percentile(67, [1, 3, 5, 9]) == 9

    f1 = percentile(67)
    assert f1([1, 3, 5, 9]) == 9

    f2 = percentile(66)
    assert f2([1, 3, 5, 9]) == 5
