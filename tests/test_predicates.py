import doctest
import pytest
from datascience import *

def test_both():
    """Both f and g."""
    p = both(above(2), below(4))
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, True, False, False]

def test_both_constant():
    """Both f and g."""
    p = both(below(3), 2)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, True, False, False, False]

def test_either():
    """Either f or g."""
    p = either(above(3), below(2))
    ps = [p(x) for x in range(1, 6)]
    assert ps == [True, False, False, True, True]

def test_either():
    """Either f or g."""
    p = either(above(3), 2)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, True, False, True, True]

def test_equal():
    """Equal to y."""
    p = equal(1)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [True, False, False, False, False]

def test_above():
    """Greater than y."""
    p = above(3)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, False, True, True]

def test_below():
    """Less than y."""
    p = not_below(4)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, False, True, True]

def test_above_or_equal():
    """Greater than or equal to y."""
    p = above_or_equal(4)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, False, True, True]

def test_below_or_equal():
    """Less than or equal to y."""
    p = not_below_or_equal(3)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, False, True, True]

def test_strictly_between():
    """Greater than y and less than z."""
    p = strictly_between(2, 4)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, True, False, False]

def test_between():
    """Greater than or equal to y and less than z."""
    p = between(3, 4)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, True, False, False]

def test_between_or_equal():
    """Greater than or equal to y and less than or equal to z."""
    p = between_or_equal(3, 3)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, True, False, False]
