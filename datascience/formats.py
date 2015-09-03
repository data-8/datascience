"""String formatting for table entries."""

import numpy as np

from datetime import datetime, timezone


class Formatter:
    """String formatter that truncates long values."""

    min_width = 4
    max_width = 60
    etc = ' ...'

    def __init__(self, min_width=None, max_width=None, etc=None):
        if min_width is not None:
            self.min_width = min_width
        if max_width is not None:
            self.max_width = max_width
        if etc is not None:
            self.etc = etc

    def format_column(self, label, column):
        """Return a formatting function that pads & truncates values."""
        if len(column) == 0:
            val_width = 0
        else:
            val_width = max(len(self.format_value(v)) for v in column)
        val_width = min(val_width, self.max_width)
        width = max(val_width, len(str(label)), self.min_width, len(self.etc))
        def pad(value, label=False):
            if label:
                raw = value
            else:
                raw = self.format_value(value)
            if len(raw) > width:
                prefix = raw[:width-len(self.etc)] + self.etc
            else:
                prefix = raw
            return prefix.ljust(width)
        return pad

    @staticmethod
    def format_value(value):
        """Pretty-print an arbitrary value."""
        if isinstance(value, (bool, np.bool_)):
            return str(value)
        elif isinstance(value, (int, np.integer)):
            return '{:n}'.format(value)
        elif isinstance(value, (float, np.floating)):
            return '{:g}'.format(value)
        else:
            return str(value)

    @staticmethod
    def convert(value):
        """Identity conversion (override to convert values)."""
        return value

    @property
    def converts_values(self):
        """Whether this Formatter also converts values."""
        return self.convert is not Formatter.convert


default_formatter = Formatter()


class NumberFormatter(Formatter):
    """Format numbers that may have delimiters."""

    converts_values = True

    def __init__(self, decimals=2, decimal_point='.', separator=','):
        self.decimals = decimals
        self.decimal_point = decimal_point
        self.separator = separator

    def convert(self, value):
        """Convert string 93,000.00 to float 93000.0."""
        if isinstance(value, str):
            value = value.replace(self.separator, '')
            if self.decimal_point not in value:
                return int(value)
            else:
                return float(value.replace(self.decimal_point, '.'))
        else:
            return value

    def format_value(self, value):
        if isinstance(value, (int, np.integer)):
            return ('{:' + self.separator + 'd}').format(value)
        else:
            return ('{:' + self.separator + '.' + str(self.decimals) + 'f}').format(value)


class CurrencyFormatter(Formatter):
    """Format currency and convert to float."""

    converts_values = True

    def __init__(self, symbol="$", *args, **vargs):
        super().__init__(*args, **vargs)
        assert isinstance(symbol, str)
        self.symbol = symbol

    def convert(self, value):
        """Convert string $1.25 to float 1.25."""
        assert isinstance(value, str), "Currency is not a string"
        assert value.startswith(self.symbol), "Currency does not start with " + self.symbol
        return float(value.lstrip(self.symbol))

    def format_value(self, value):
        """Format currency."""
        return self.symbol + "{0:,.2f}".format(value)


class DateFormatter(Formatter):
    """Format date & time and convert to UNIX timestamp."""

    converts_values = True

    def __init__(self, format="%Y-%m-%d %H:%M:%S.%f", *args, **vargs):
        super().__init__(*args, **vargs)
        assert isinstance(format, str)
        self.format = format

    def convert(self, value):
        """Convert 2015-08-03 to a Unix timestamp int."""
        return datetime.strptime(value, self.format).timestamp()

    def format_value(self, value):
        """Format timestamp as a string."""
        return datetime.fromtimestamp(value).strftime(self.format)


class PercentFormatter(Formatter):
    """Format a number as a percentage."""

    converts_values = False

    def __init__(self, decimals=2, *args, **vargs):
        assert isinstance(decimals, int)
        self.decimals = decimals

    def format_value(self, value):
        """Format number as percentage."""
        return ('{:.' + str(self.decimals) + '%}').format(value)

