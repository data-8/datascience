import pytest

from datascience import *

def test_percentile():
    assert percentile([1, 3, 5, 9], 0) == 1
    assert percentile([1, 3, 5, 9], 33) == 3
    assert percentile([1, 3, 5, 9], 34) == 5
    assert percentile([1, 3, 5, 9], 66) == 5
    assert percentile([1, 3, 5, 9], 67) == 9
