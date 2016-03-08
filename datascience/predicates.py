"""Predicate functions."""

all = ['both', 'either', 'Is']

class Is:
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

Is.not_equal_to = _not(Is.equal_to)
Is.not_above = Is.below_or_equal_to
Is.not_below = Is.above_or_equal_to
Is.not_below_or_equal_to = Is.above
Is.not_above_or_equal_to = Is.below
Is.not_strictly_between = _not(Is.strictly_between)
Is.not_between = _not(Is.between)
Is.not_between_or_equal_to = _not(Is.between_or_equal_to)

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
        return Is.equal_to(f)
    else:
        return f

