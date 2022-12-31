import doctest
import pytest
from datascience import predicates
from datascience import *

def test_both():
    """Both f and g."""
    p = are.above(2) & are.below(4)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, True, False, False]

def test_either():
    """Either f or g."""
    p = are.above(3) | are.below(2)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [True, False, False, True, True]

def test_equal_to():
    """Equal to y."""
    p = are.equal_to(1)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [True, False, False, False, False]

def test_above():
    """Greater than y."""
    p = are.above(3)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, False, True, True]

def test_below():
    """Less than y."""
    p = are.not_below(4)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, False, True, True]

def test_above_or_equal_to():
    """Greater than or equal to y."""
    p = are.above_or_equal_to(4)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, False, True, True]

def test_below_or_equal_to():
    """Less than or equal to y."""
    p = are.not_below_or_equal_to(3)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, False, True, True]

def test_strictly_between():
    """Greater than y and less than z."""
    p = are.strictly_between(2, 4)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, True, False, False]

def test_between():
    """Greater than or equal to y and less than z."""
    p = are.between(3, 4)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, True, False, False]

def test_between_or_equal_to():
    """Greater than or equal to y and less than or equal to z."""
    p = are.between_or_equal_to(3, 3)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, True, False, False]

#############
#  Aliases  #
#############

def test_greater_than_and_less_than():
    """Both f and g."""
    p = are.greater_than(2) & are.less_than(4)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, True, False, False]

def test_greater_than_or_less_than():
    """Either f or g."""
    p = are.greater_than(3) | are.less_than(2)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [True, False, False, True, True]

def test_greater_than():
    """Greater than y."""
    p = are.greater_than(3)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, False, True, True]

def test_less_than():
    """Less than y."""
    p = are.not_less_than(4)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, False, True, True]

def test_greater_than_or_equal_to():
    """Greater than or equal to y."""
    p = are.greater_than_or_equal_to(4)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, False, True, True]

def test_less_than_or_equal_to():
    """Less than or equal to y."""
    p = are.not_less_than_or_equal_to(3)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, False, True, True]


############
# Doctests #
############


def test_doctests():
    results = doctest.testmod(predicates,
                              optionflags=doctest.NORMALIZE_WHITESPACE)
    assert results.failed == 0
