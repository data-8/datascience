"""Predicate functions."""

all = ['both', 'either', 'are']

class are:
    """Predicate class."""

    @staticmethod
    def equal_to(y):
        """Equal to y."""
        return lambda x: x == y

    @staticmethod
    def above(y):
        """Greater than y."""
        return lambda x: x > y

    @staticmethod
    def below(y):
        """Less than y."""
        return lambda x: x < y

    @staticmethod
    def above_or_equal_to(y):
        """Greater than or equal to y."""
        return lambda x: x >= y

    @staticmethod
    def below_or_equal_to(y):
        """Less than or equal to y."""
        return lambda x: x <= y

    @staticmethod
    def strictly_between(y, z):
        """Greater than y and less than z."""
        return lambda x: y < x < z

    @staticmethod
    def between(y, z):
        """Greater than or equal to y and less than z."""
        return lambda x: y <= x < z

    @staticmethod
    def between_or_equal_to(y, z):
        """Greater than or equal to y and less than or equal to z."""
        return lambda x: y <= x <= z

############
# Negation #
############

def _not(f):
    return lambda x: not f(x)

are.not_equal_to = _not(are.equal_to)
are.not_above = are.below_or_equal_to
are.not_below = are.above_or_equal_to
are.not_below_or_equal_to = are.above
are.not_above_or_equal_to = are.below
are.not_strictly_between = _not(are.strictly_between)
are.not_between = _not(are.between)
are.not_between_or_equal_to = _not(are.between_or_equal_to)

###############
# Combination #
###############

def both(f, g):
    """Both f and g."""
    f, g = _wrap_equal_to(f), _wrap_equal_to(g)
    return lambda x: f(x) and g(x)

def either(f, g):
    """Either f or g."""
    f, g = _wrap_equal_to(f), _wrap_equal_to(g)
    return lambda x: f(x) or g(x)

def _wrap_equal_to(f):
    if not callable(f):
        return are.equal_to(f)
    else:
        return f

are.both = both
are.either = either
