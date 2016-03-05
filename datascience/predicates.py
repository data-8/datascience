"""Predicate functions."""

def both(f, g):
    """Both f and g."""
    f, g = _wrap_equal(f), _wrap_equal(g)
    return lambda x: f(x) and g(x)

def either(f, g):
    """Either f or g."""
    f, g = _wrap_equal(f), _wrap_equal(g)
    return lambda x: f(x) or g(x)

def equal(y):
    """Equal to y."""
    return lambda x: x == y

def above(y):
    """Greater than y."""
    return lambda x: x > y

def below(y):
    """Less than y."""
    return lambda x: x < y

def above_or_equal(y):
    """Greater than or equal to y."""
    return lambda x: x >= y

def below_or_equal(y):
    """Less than or equal to y."""
    return lambda x: x <= y

def strictly_between(y, z):
    """Greater than y and less than z."""
    return lambda x: y < x < z

def between(y, z):
    """Greater than or equal to y and less than z."""
    return lambda x: y <= x < z

def between_or_equal(y, z):
    """Greater than or equal to y and less than or equal to z."""
    return lambda x: y <= x <= z

def _not(f):
    return lambda x: not f(x)

def _wrap_equal(f):
    if not callable(f):
        return equal(f)
    else:
        return f

not_equal = _not(equal)
not_above = below_or_equal
not_below = above_or_equal
not_below_or_equal = above
not_above_or_equal = below
not_strictly_between = _not(strictly_between)
not_between = _not(between)
not_between_or_equal = _not(between_or_equal)


