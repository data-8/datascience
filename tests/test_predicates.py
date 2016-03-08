import doctest
import pytest
from datascience import *

def test_both():
    """Both f and g."""
    p = both(Is.above(2), Is.below(4))
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, True, False, False]

def test_both_constant():
    """Both f and g."""
    p = both(Is.below(3), 2)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, True, False, False, False]

def test_either():
    """Either f or g."""
    p = either(Is.above(3), Is.below(2))
    ps = [p(x) for x in range(1, 6)]
    assert ps == [True, False, False, True, True]

def test_either():
    """Either f or g."""
    p = either(Is.above(3), 2)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, True, False, True, True]

def test_equal_to():
    """Equal to y."""
    p = Is.equal_to(1)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [True, False, False, False, False]

def test_above():
    """Greater than y."""
    p = Is.above(3)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, False, True, True]

def test_below():
    """Less than y."""
    p = Is.not_below(4)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, False, True, True]

def test_above_or_equal_to():
    """Greater than or equal to y."""
    p = Is.above_or_equal_to(4)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, False, True, True]

def test_below_or_equal_to():
    """Less than or equal to y."""
    p = Is.not_below_or_equal_to(3)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, False, True, True]

def test_strictly_between():
    """Greater than y and less than z."""
    p = Is.strictly_between(2, 4)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, True, False, False]

def test_between():
    """Greater than or equal to y and less than z."""
    p = Is.between(3, 4)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, True, False, False]

def test_between_or_equal_to():
    """Greater than or equal to y and less than or equal to z."""
    p = Is.between_or_equal_to(3, 3)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, True, False, False]
