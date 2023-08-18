import doctest
import pytest
from datascience import predicates
from datascience import *

def test_both():
    """Both f and g."""
    p = are.above(2) & are.below(4)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, True, False, False]

def test_tautology():
    p = are.above(3) | are.below(4)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [True, True, True, True, True]

def test_contradiction():
    p = are.above(3) & are.below(3)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [False, False, False, False, False]

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

def test_not_equal_to():
    """Greater than or equal to y and less than or equal to z."""
    p = are.not_equal_to(5)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [True, True, True, True, False]

def test_not():
    """Not f."""
    not_f = predicates._not(lambda x: 2*x)
    assert not_f(5) == -10

def test_xor():
    """XOR f and g."""
    p = are.above(3) ^ are.below(2)
    ps = [p(x) for x in range(1, 6)]
    assert ps == [True, False, False, True, True]

def test_containing():
    """Containing substring"""
    p = are.containing("L")
    string_list = ["Lemon","LidL","like","Amsterdam","buLk"]
    ps = [p(x) for x in string_list]
    assert ps == [True, True, False, False, True]

def test_not_containing():
    """Not containing substring"""
    p = are.containing("f")
    string_list = ["lime","Spark","Friend","fire","culf"]
    ps = [p(x) for x in string_list]
    assert ps == [True, True, False, False, True]

############
# Doctests #
############


def test_doctests():
    results = doctest.testmod(predicates,
                              optionflags=doctest.NORMALIZE_WHITESPACE)
    assert results.failed == 0
