"""Tables are sequences of labeled columns."""

__all__ = ['Table']

import abc
import collections
import collections.abc
import functools
import inspect
import itertools
import numbers
import urllib.parse
import warnings

import numpy as np
import matplotlib
matplotlib.use('agg', warn=False)
import matplotlib.pyplot as plt
import pandas
import IPython

import datascience.maps as _maps
import datascience.formats as _formats
import datascience.util as _util
import datascience.predicates as _predicates

class Table(collections.abc.MutableMapping):
    """A sequence of string-labeled columns."""
    plots = []

    def __init__(self, labels=None, _deprecated=None, *, formatter=_formats.default_formatter):
        """Create an empty table with column labels.

        >>> tiles = Table(['letter', 'count', 'points'])
        >>> tiles
        letter | count | points

        Args:
            ``labels`` (list of strings): The column labels.

            ``formatter`` (Formatter): An instance of :class:`Formatter` that
                formats the columns' values.
        """
        self._columns = collections.OrderedDict()
        self._formats = dict()
        self.formatter = formatter

        if _deprecated is not None:
            warnings.warn("Two-argument __init__ is deprecated. Use Table().with_columns(...)", FutureWarning)
            columns, labels = labels, _deprecated
            columns = columns if columns is not None else []
            labels = labels if labels is not None else []
            assert len(labels) == len(columns), 'label/column number mismatch'
        else:
            labels = labels if labels is not None else []
            columns = [[] for _ in labels]

        self._num_rows = 0 if len(columns) is 0 else len(columns[0])

        # Add each column to table
        for column, label in zip(columns, labels):
            self[label] = column

        self.take = _RowTaker(self)
        self.exclude = _RowExcluder(self)

    # Deprecated
    @classmethod
    def empty(cls, labels=None):
        """Create an empty table. Column labels are optional. [Deprecated]

        Args:
            ``labels`` (None or list): If ``None``, a table with 0
                columns is created.
                If a list, each element is a column label in a table with
                0 rows.

        Returns:
            A new instance of ``Table``.
        """
        warnings.warn("Table.empty(labels) is deprecated. Use Table(labels)", FutureWarning)
        if labels is None:
            return cls()
        values = [[] for label in labels]
        return cls(values, labels)

    # Deprecated
    @classmethod
    def from_rows(cls, rows, labels):
        """Create a table from a sequence of rows (fixed-length sequences). [Deprecated]"""
        warnings.warn("Table.from_rows is deprecated. Use Table(labels).with_rows(...)", FutureWarning)
        return cls(labels).with_rows(rows)

    @classmethod
    def from_records(cls, records):
        """Create a table from a sequence of records (dicts with fixed keys)."""
        if not records:
            return cls()
        labels = sorted(list(records[0].keys()))
        columns = [[rec[label] for rec in records] for label in labels]
        return cls().with_columns(zip(labels, columns))

    # Deprecated
    @classmethod
    def from_columns_dict(cls, columns):
        """Create a table from a mapping of column labels to column values. [Deprecated]"""
        warnings.warn("Table.from_columns_dict is deprecated. Use Table().with_columns(...)", FutureWarning)
        return cls().with_columns(columns.items())

    @classmethod
    def read_table(cls, filepath_or_buffer, *args, **vargs):
        """Read a table from a file or web address.

        filepath_or_buffer -- string or file handle / StringIO; The string
                              could be a URL. Valid URL schemes include http,
                              ftp, s3, and file.
        """
        # Look for .csv at the end of the path; use "," as a separator if found
        try:
            path = urllib.parse.urlparse(filepath_or_buffer).path
            if 'data8.berkeley.edu' in filepath_or_buffer:
                raise ValueError('data8.berkeley.edu requires authentication, '
                                 'which is not supported.')
        except AttributeError:
            path = filepath_or_buffer

        try:
            if 'sep' not in vargs and path.endswith('.csv'):
                vargs['sep'] = ','
        except AttributeError:
            pass
        df = pandas.read_table(filepath_or_buffer, *args, **vargs)
        return cls.from_df(df)

    def _with_columns(self, columns):
        """Create a table from a sequence of columns, copying column labels."""
        table = type(self)()
        for label, column in zip(self.labels, columns):
            self._add_column_and_format(table, label, column)
        return table

    def _add_column_and_format(self, table, label, column):
        """Add a column to table, copying the formatter from self."""
        label = self._as_label(label)
        table[label] = column
        if label in self._formats:
            table._formats[label] = self._formats[label]

    @classmethod
    def from_df(cls, df):
        """Convert a Pandas DataFrame into a Table."""
        t = cls()
        labels = df.columns
        for label in df.columns:
            t.append_column(label, df[label])
        return t

    @classmethod
    def from_array(cls, arr):
        """Convert a structured NumPy array into a Table."""
        return cls().with_columns([(f, arr[f]) for f in arr.dtype.names])

    #################
    # Magic Methods #
    #################

    def __getitem__(self, index_or_label):
        label = self._as_label(index_or_label)
        return self.column(label)

    def __setitem__(self, label, values):
        self.append_column(label, values)

    def __delitem__(self, label):
        del self._columns[label]
        if label in self._formats:
            del self._formats[label]

    def __len__(self):
        return len(self._columns)

    def __iter__(self):
        return iter(self.labels)

    # Deprecated
    def __getattr__(self, attr):
        """Return a method that applies to all columns or a table of attributes. [Deprecated]

        E.g., t.sum() on a Table will return a table with the sum of each column.
        """

        if self.columns and all(hasattr(c, attr) for c in self.columns):
            warnings.warn("Implicit column method lookup is deprecated.", FutureWarning)
            attrs = [getattr(c, attr) for c in self.columns]
            if all(callable(attr) for attr in attrs):
                @functools.wraps(attrs[0])
                def method(*args, **vargs):
                    """Create a table from the results of calling attrs."""
                    columns = [attr(*args, **vargs) for attr in attrs]
                    return self._with_columns(columns)
                return method
            else:
                return self._with_columns([[attr] for attr in attrs])
        else:
            msg = "'{0}' object has no attribute '{1}'".format(type(self).__name__, attr)
            raise AttributeError(msg)

    ####################
    # Accessing Values #
    ####################

    @property
    def num_rows(self):
        """Number of rows."""
        return self._num_rows

    @property
    def rows(self):
        """Return a view of all rows."""
        return self.Rows(self)

    def row(self, index):
        """Return a row."""
        return self.rows[index]

    @property
    def labels(self):
        """Return a tuple of column labels."""
        return tuple(self._columns.keys())

    # Deprecated
    @property
    def column_labels(self):
        """Return a tuple of column labels. [Deprecated]"""
        warnings.warn("column_labels is deprecated; use labels", FutureWarning)
        return self.labels

    @property
    def num_columns(self):
        """Number of columns."""
        return len(self.labels)

    @property
    def columns(self):
        return tuple(self._columns.values())

    def column(self, index_or_label):
        """Return the values of a column as an array.

        table.column(label) is equivalent to table[label].

        >>> tiles = Table().with_columns([
        ...     'letter', ['c', 'd'],
        ...     'count',  [2, 4],
        ... ])
        >>> list(tiles.column('letter'))
        ['c', 'd']
        >>> tiles.column(1)
        array([2, 4])

        Args:
            label (int or str): The index or label of a column

        Returns:
            An instance of ``numpy.array``.
        """
        return self._columns[self._as_label(index_or_label)]

    # Deprecated
    def values(self, label):
        """Returns the values of a column as an array. [Deprecated]"""
        warnings.warn("values is deprecated; use column", FutureWarning)
        return self.column(label)

    def column_index(self, column_label):
        """Return the index of a column."""
        return self.labels.index(column_label)

    def apply(self, fn, column_label=None):
        """Returns an array where ``fn`` is applied to each set of elements
        by row from the specified columns in ``column_label``. If no
        column_label is specified, then each row is passed to fn.

        Args:
            ``fn`` (function): The function to be applied to elements specified
                by ``column_label``.
            ``column_label`` (single string or list of strings): Names of
                columns to be passed into function ``fn``. Length must match
                number of elements ``fn`` takes.

        Raises:
            ``ValueError``: column name in ``column_label`` is not an existing
                column in the table.

        Returns:
            A numpy array consisting of results of applying ``fn`` to elements
            specified by ``column_label`` in each row.

        >>> t = Table().with_columns([
        ...     'letter', ['a', 'b', 'c', 'z'],
        ...     'count',  [9, 3, 3, 1],
        ...     'points', [1, 2, 2, 10]])
        >>> t
        letter | count | points
        a      | 9     | 1
        b      | 3     | 2
        c      | 3     | 2
        z      | 1     | 10
        >>> t.apply(lambda x: x - 1, 'points')
        array([0, 1, 1, 9])
        >>> t.apply(lambda x, y: x * y, ['count', 'points'])
        array([ 9,  6,  6, 10])

        Whole rows are passed to the function if no columns are specified.

        >>> t.apply(lambda row: row.item('count') * 2)
        array([18,  6,  6,  2])
        """
        if column_label is None:
            return np.array([fn(row) for row in self.rows])
        else:
            rows = zip(*self.select(column_label).columns)
            return np.array([fn(*row) for row in rows])

    ############
    # Mutation #
    ############

    def set_format(self, column_label_or_labels, formatter):
        """Set the format of a column."""
        if inspect.isclass(formatter) and issubclass(formatter, _formats.Formatter):
            formatter = formatter()
        for label in self._as_labels(column_label_or_labels):
            if callable(formatter):
                self._formats[label] = lambda v, label: v if label else str(formatter(v))
            elif isinstance(formatter, _formats.Formatter):
                if formatter.converts_values:
                    self[label] = self.apply(formatter.convert, label)
                column = self[label]
                self._formats[label] = formatter.format_column(label, column)
            else:
                raise Exception('Expected Formatter or function: ' + str(formatter))
        return self

    def move_to_start(self, column_label):
        """Move a column to the first in order."""
        self._columns.move_to_end(column_label, last=False)
        return self

    def move_to_end(self, column_label):
        """Move a column to the last in order."""
        self._columns.move_to_end(column_label)
        return self

    def append(self, row_or_table):
        """Append a row or all rows of a table. An appended table must have all
        columns of self."""
        if not row_or_table:
            return
        if isinstance(row_or_table, Table):
            t = row_or_table
            columns = list(t.select(self.labels)._columns.values())
            n = t.num_rows
        else:
            columns, n = [[value] for value in row_or_table], 1
        for i, column in enumerate(self._columns):
            if self.num_rows:
                self._columns[column] = np.append(self[column], columns[i])
            else:
                self._columns[column] = np.array(columns[i])
        self._num_rows += n
        return self

    def append_column(self, label, values):
        """Appends a column to the table or replaces a column.

        ``__setitem__`` is aliased to this method:
            ``table.append_column('new_col', [1, 2, 3])`` is equivalent to
            ``table['new_col'] = [1, 2, 3]``.

        Args:
            ``label`` (str): The label of the new column.

            ``values`` (single value or list/array): If a single value, every
                value in the new column is ``values``.

                If a list or array, the new column contains the values in
                ``values``, which must be the same length as the table.

        Returns:
            Original table with new or replaced column

        Raises:
            ``ValueError``: If
                - ``label`` is not a string.
                - ``values`` is a list/array and does not have the same length
                  as the number of rows in the table.

        >>> table = Table().with_columns([
        ...     'letter', ['a', 'b', 'c', 'z'],
        ...     'count',  [9, 3, 3, 1],
        ...     'points', [1, 2, 2, 10]])
        >>> table
        letter | count | points
        a      | 9     | 1
        b      | 3     | 2
        c      | 3     | 2
        z      | 1     | 10
        >>> table.append_column('new_col1', [10, 20, 30, 40])
        >>> table
        letter | count | points | new_col1
        a      | 9     | 1      | 10
        b      | 3     | 2      | 20
        c      | 3     | 2      | 30
        z      | 1     | 10     | 40
        >>> table.append_column('new_col2', 'hello')
        >>> table
        letter | count | points | new_col1 | new_col2
        a      | 9     | 1      | 10       | hello
        b      | 3     | 2      | 20       | hello
        c      | 3     | 2      | 30       | hello
        z      | 1     | 10     | 40       | hello
        >>> table.append_column(123, [1, 2, 3, 4])
        Traceback (most recent call last):
            ...
        ValueError: The column label must be a string, but a int was given
        >>> table.append_column('bad_col', [1, 2])
        Traceback (most recent call last):
            ...
        ValueError: Column length mismatch. New column does not have the same number of rows as table.
        """
        # TODO(sam): Allow append_column to take in a another table, copying
        # over formatter as needed.
        if not isinstance(label, str):
            raise ValueError('The column label must be a string, but a '
                '{} was given'.format(label.__class__.__name__))

        if not isinstance(values, np.ndarray):
            # Coerce a single value to a sequence
            if not _is_non_string_iterable(values):
                values = [values] * max(self.num_rows, 1)
            values = np.array(tuple(values))

        if self.num_rows != 0 and len(values) != self.num_rows:
            raise ValueError('Column length mismatch. New column does not have '
                             'the same number of rows as table.')
        else:
            self._num_rows = len(values)

        self._columns[label] = values

    def relabel(self, column_label, new_label):
        """Change the labels of columns specified by ``column_label`` to
        labels in ``new_label``.

        Args:
            ``column_label`` (single str or list/array of str): The label(s) of
                columns to be changed. Must be str.

            ``new_label`` (single str or list/array of str): The new label(s) of
                columns to be changed. Must be str.

                Number of elements must match number of elements in
                ``column_label``.

        Returns:
            Original table with modified labels

        >>> table = Table().with_columns([
        ...     'points', (1, 2, 3),
        ...     'id',     (12345, 123, 5123)])
        >>> table.relabel('id', 'yolo')
        points | yolo
        1      | 12345
        2      | 123
        3      | 5123
        >>> table.relabel(['points', 'yolo'], ['red', 'blue'])
        red  | blue
        1    | 12345
        2    | 123
        3    | 5123
        >>> table.relabel(['red', 'green', 'blue'], ['cyan', 'magenta', 'yellow', 'key'])
        Traceback (most recent call last):
            ...
        ValueError: Invalid arguments. column_label and new_label must be of equal length.
        >>> table.relabel(['red', 'blue'], ['blue', 'red'])
        blue | red
        1    | 12345
        2    | 123
        3    | 5123
        """
        if isinstance(column_label, numbers.Integral):
            column_label = self._as_label(column_label)
        if isinstance(column_label, str) and isinstance(new_label, str):
            column_label, new_label = [column_label], [new_label]
        if len(column_label) != len(new_label):
            raise ValueError('Invalid arguments. column_label and new_label '
                'must be of equal length.')
        old_to_new = dict(zip(column_label, new_label)) # maps old labels to new ones
        for label in column_label:
            if not (label in self.labels):
                raise ValueError('Invalid labels. Column labels must '
                'already exist in table in order to be replaced.')
        rewrite = lambda s: old_to_new[s] if s in old_to_new else s
        columns = [(rewrite(s), c) for s, c in self._columns.items()]
        self._columns = collections.OrderedDict(columns)
        for label in self._formats:
            # TODO(denero) Error when old and new columns share a name
            if label in column_label:
                formatter = self._formats.pop(label)
                self._formats[old_to_new[label]] = formatter
        return self


    ##################
    # Transformation #
    ##################

    def copy(self, *, shallow=False):
        """Return a copy of a Table."""
        table = type(self)()
        for label in self.labels:
            if shallow:
                column = self[label]
            else:
                column = np.copy(self[label])
            self._add_column_and_format(table, label, column)
        return table

    def select(self, column_label_or_labels):
        """Return a Table with selected column or columns by label or index.

        Args:
            ``column_label_or_labels`` (string or list of strings): The header
            names or indices of the columns to be selected. ``column_label_or_labels`` must
            be an existing header name, or a valid column index.

        Returns:
            An instance of ``Table`` containing only selected columns.

        >>> t = Table().with_columns([
        ...     'burgers',  ['cheeseburger', 'hamburger', 'veggie burger'],
        ...     'prices',   [6, 5, 5],
        ...     'calories', [743, 651, 582]])
        >>> t
        burgers       | prices | calories
        cheeseburger  | 6      | 743
        hamburger     | 5      | 651
        veggie burger | 5      | 582
        >>> t.select(['burgers', 'calories'])
        burgers       | calories
        cheeseburger  | 743
        hamburger     | 651
        veggie burger | 582
        >>> t.select('prices')
        prices
        6
        5
        5
        >>> t.select(1)
        prices
        6
        5
        5
        >>> t.select([2, 0])
        calories | burgers
        743      | cheeseburger
        651      | hamburger
        582      | veggie burger
        """
        labels = self._as_labels(column_label_or_labels)
        table = Table()
        for label in labels:
            self._add_column_and_format(table, label, np.copy(self[label]))
        return table

    # These, along with a snippet below, are necessary for Sphinx to
    # correctly load the `take` and `exclude` docstrings.  The definitions
    # will be over-ridden during class instantiation.
    def take(self):
        raise NotImplementedError()

    def exclude(self):
        raise NotImplementedError()

    def drop(self, column_label_or_labels):
        """Return a Table with only columns other than selected label or labels.

        Args:
            ``column_label_or_labels`` (string or list of strings): The header
            names or indices of the columns to be dropped. ``column_label_or_labels`` must
            be an existing header name, or a valid column index.

        Returns:
            An instance of ``Table`` with given columns removed.

        >>> t = Table().with_columns([
        ...     'burgers',  ['cheeseburger', 'hamburger', 'veggie burger'],
        ...     'prices',   [6, 5, 5],
        ...     'calories', [743, 651, 582]])
        >>> t
        burgers       | prices | calories
        cheeseburger  | 6      | 743
        hamburger     | 5      | 651
        veggie burger | 5      | 582
        >>> t.drop('prices')
        burgers       | calories
        cheeseburger  | 743
        hamburger     | 651
        veggie burger | 582
        >>> t.drop(['burgers', 'calories'])
        prices
        6
        5
        5
        >>> t.drop([0, 2])
        prices
        6
        5
        5
        >>> t.drop(1)
        burgers       | calories
        cheeseburger  | 743
        hamburger     | 651
        veggie burger | 582
        """
        exclude = _as_labels(column_label_or_labels)
        return self.select([c for (i, c) in enumerate(self.labels) if i not in exclude and c not in exclude])

    def where(self, column_or_label, value_or_predicate=None, other=None):
        """Return a Table of rows for which the column is ``value`` or a non-zero value.

        If ``column_or_label`` contains Boolean values, returns rows corresponding to True.

        Args:
            ``column_or_label``: The header name of a column in the table or an array.

            ``value_or_predicate``: Value to compare to items in column
            or function to apply to items in column.

        Returns:
            An instance of ``Table`` containing rows for which the ``column_or_label``
            column or ``column_or_label`` itself is non-zero or True, or is equal to ``value``,
            if provided.

        >>> marbles = Table().with_columns([
        ...    "Color", ["Red", "Green", "Blue", "Red", "Green", "Green"],
        ...    "Shape", ["Round", "Rectangular", "Rectangular", "Round", "Rectangular", "Round"],
        ...    "Amount", [4, 6, 12, 7, 9, 2],
        ...    "Price", [1.30, 1.20, 2.00, 1.75, 1.40, 3.00]])
        >>> marbles
        Color | Shape       | Amount | Price
        Red   | Round       | 4      | 1.3
        Green | Rectangular | 6      | 1.2
        Blue  | Rectangular | 12     | 2
        Red   | Round       | 7      | 1.75
        Green | Rectangular | 9      | 1.4
        Green | Round       | 2      | 3
        >>> marbles.where("Shape", "Round")
        Color | Shape | Amount | Price
        Red   | Round | 4      | 1.3
        Red   | Round | 7      | 1.75
        Green | Round | 2      | 3
        >>> marbles.where(marbles.column("Shape") == "Round") # equivalent to previous example
        Color | Shape | Amount | Price
        Red   | Round | 4      | 1.3
        Red   | Round | 7      | 1.75
        Green | Round | 2      | 3
        >>> marbles.where(marbles.column("Price") > 1.5)
        Color | Shape       | Amount | Price
        Blue  | Rectangular | 12     | 2
        Red   | Round       | 7      | 1.75
        Green | Round       | 2      | 3

        You can also use predicates to simplify single-column comparisons.

        >>> from datascience.predicates import are
        >>> marbles.where("Price", are.above(1.5)) # equivalent to previous example
        Color | Shape       | Amount | Price
        Blue  | Rectangular | 12     | 2
        Red   | Round       | 7      | 1.75
        Green | Round       | 2      | 3

        And apply some predicates to compare columns.

        >>> marbles.where("Price", are.above, "Amount")
        Color | Shape | Amount | Price
        Green | Round | 2      | 3
        """
        column = self._get_column(column_or_label)
        if other is not None:
            assert callable(value_or_predicate), "Predicate required for 3-arg where"
            predicate = value_or_predicate
            other = self._get_column(other)
            column = [predicate(y)(x) for x, y in zip(column, other)]
        elif value_or_predicate is not None:
            if not callable(value_or_predicate):
                predicate = _predicates.are.equal_to(value_or_predicate)
            else:
                predicate = value_or_predicate
            column = [predicate(x) for x in column]
        return self.take(np.nonzero(column)[0])

    def sort(self, column_or_label, descending=False, distinct=False):
        """Return a Table of rows sorted according to the values in a column.

        Args:
            ``column_or_label``: the column whose values are used for sorting.

            ``descending``: if True, sorting will be in descending, rather than
            ascending order.

            ``distinct``: if True, repeated values in ``column_or_label`` will be omitted.

        Returns:
            An instance of ``Table`` containing rows sorted based on the values in ``column_or_label``.

        >>> marbles = Table().with_columns([
        ...    "Color", ["Red", "Green", "Blue", "Red", "Green", "Green"],
        ...    "Shape", ["Round", "Rectangular", "Rectangular", "Round", "Rectangular", "Round"],
        ...    "Amount", [4, 6, 12, 7, 9, 2],
        ...    "Price", [1.30, 1.30, 2.00, 1.75, 1.40, 1.00]])
        >>> marbles
        Color | Shape       | Amount | Price
        Red   | Round       | 4      | 1.3
        Green | Rectangular | 6      | 1.3
        Blue  | Rectangular | 12     | 2
        Red   | Round       | 7      | 1.75
        Green | Rectangular | 9      | 1.4
        Green | Round       | 2      | 1
        >>> marbles.sort("Amount")
        Color | Shape       | Amount | Price
        Green | Round       | 2      | 1
        Red   | Round       | 4      | 1.3
        Green | Rectangular | 6      | 1.3
        Red   | Round       | 7      | 1.75
        Green | Rectangular | 9      | 1.4
        Blue  | Rectangular | 12     | 2
        >>> marbles.sort("Amount", descending = True)
        Color | Shape       | Amount | Price
        Blue  | Rectangular | 12     | 2
        Green | Rectangular | 9      | 1.4
        Red   | Round       | 7      | 1.75
        Green | Rectangular | 6      | 1.3
        Red   | Round       | 4      | 1.3
        Green | Round       | 2      | 1
        >>> marbles.sort(3) # the Price column
        Color | Shape       | Amount | Price
        Green | Round       | 2      | 1
        Red   | Round       | 4      | 1.3
        Green | Rectangular | 6      | 1.3
        Green | Rectangular | 9      | 1.4
        Red   | Round       | 7      | 1.75
        Blue  | Rectangular | 12     | 2
        >>> marbles.sort(3, distinct = True)
        Color | Shape       | Amount | Price
        Green | Round       | 2      | 1
        Red   | Round       | 4      | 1.3
        Green | Rectangular | 9      | 1.4
        Red   | Round       | 7      | 1.75
        Blue  | Rectangular | 12     | 2
        """
        column = self._get_column(column_or_label)
        if distinct:
            _, row_numbers = np.unique(column, return_index=True)
        else:
            row_numbers = np.argsort(column, axis=0, kind='mergesort')
        assert (row_numbers < self.num_rows).all(), row_numbers
        if descending:
            row_numbers = np.array(row_numbers[::-1])
        return self.take(row_numbers)

    def group(self, column_or_label, collect=None):
        """Group rows by unique values in a column; count or aggregate others.

        Args:
            ``column_or_label``: values to group (column label or index, or array)

            ``collect``: a function applied to values in other columns for each group

        Returns:
            A Table with each row corresponding to a unique value in ``column_or_label``,
            where the first column contains the unique values from ``column_or_label``, and the
            second contains counts for each of the unique values. If ``collect`` is
            provided, a Table is returned with all original columns, each containing values
            calculated by first grouping rows according to ``column_or_label``, then applying
            ``collect`` to each set of grouped values in the other columns.

        Note:
            The grouped column will appear first in the result table. If ``collect`` does not
            accept arguments with one of the column types, that column will be empty in the resulting
            table.

        >>> marbles = Table().with_columns([
        ...    "Color", ["Red", "Green", "Blue", "Red", "Green", "Green"],
        ...    "Shape", ["Round", "Rectangular", "Rectangular", "Round", "Rectangular", "Round"],
        ...    "Amount", [4, 6, 12, 7, 9, 2],
        ...    "Price", [1.30, 1.30, 2.00, 1.75, 1.40, 1.00]])
        >>> marbles
        Color | Shape       | Amount | Price
        Red   | Round       | 4      | 1.3
        Green | Rectangular | 6      | 1.3
        Blue  | Rectangular | 12     | 2
        Red   | Round       | 7      | 1.75
        Green | Rectangular | 9      | 1.4
        Green | Round       | 2      | 1
        >>> marbles.group("Color") # just gives counts
        Color | count
        Blue  | 1
        Green | 3
        Red   | 2
        >>> marbles.group("Color", max) # takes the max of each grouping, in each column
        Color | Shape max   | Amount max | Price max
        Blue  | Rectangular | 12         | 2
        Green | Round       | 9          | 1.4
        Red   | Round       | 7          | 1.75
        >>> marbles.group("Shape", sum) # sum doesn't make sense for strings
        Shape       | Color sum | Amount sum | Price sum
        Rectangular |           | 27         | 4.7
        Round       |           | 13         | 4.05
        """
        self = self.copy(shallow=True)
        collect = _zero_on_type_error(collect)

        # Remove column used for grouping
        column = self._get_column(column_or_label)
        if isinstance(column_or_label, str) or isinstance(column_or_label, numbers.Integral):
            column_label = self._as_label(column_or_label)
            del self[column_label]
        else:
            column_label = self._unused_label('group')

        # Group by column
        groups = self.index_by(column)
        keys = sorted(groups.keys())

        # Generate grouped columns
        if collect is None:
            labels = [column_label, 'count' if column_label != 'count' else self._unused_label('count')]
            columns = [keys, [len(groups[k]) for k in keys]]
        else:
            columns, labels = [], []
            for i, label in enumerate(self.labels):
                labels.append(_collected_label(collect, label))
                c = [collect(np.array([row[i] for row in groups[k]])) for k in keys]
                columns.append(c)

        grouped = type(self)().with_columns(zip(labels, columns))
        assert column_label == self._unused_label(column_label)
        grouped[column_label] = keys
        grouped.move_to_start(column_label)
        return grouped

    def groups(self, labels, collect=None):
        """Group rows by multiple columns, count or aggregate others.

        Args:
            ``labels``: list of column names (or indices) to group on

            ``collect``: a function applied to values in other columns for each group

        Returns: A Table with each row corresponding to a unique combination of values in
            the columns specified in ``labels``, where the first columns are those
            specified in ``labels``, followed by a column of counts for each of the unique
            values. If ``collect`` is provided, a Table is returned with all original
            columns, each containing values calculated by first grouping rows according to
            to values in the ``labels`` column, then applying ``collect`` to each set of
            grouped values in the other columns.

        Note:
            The grouped columns will appear first in the result table. If ``collect`` does not
            accept arguments with one of the column types, that column will be empty in the resulting
            table.

        >>> marbles = Table().with_columns([
        ...    "Color", ["Red", "Green", "Blue", "Red", "Green", "Green"],
        ...    "Shape", ["Round", "Rectangular", "Rectangular", "Round", "Rectangular", "Round"],
        ...    "Amount", [4, 6, 12, 7, 9, 2],
        ...    "Price", [1.30, 1.30, 2.00, 1.75, 1.40, 1.00]])
        >>> marbles
        Color | Shape       | Amount | Price
        Red   | Round       | 4      | 1.3
        Green | Rectangular | 6      | 1.3
        Blue  | Rectangular | 12     | 2
        Red   | Round       | 7      | 1.75
        Green | Rectangular | 9      | 1.4
        Green | Round       | 2      | 1
        >>> marbles.groups(["Color", "Shape"])
        Color | Shape       | count
        Blue  | Rectangular | 1
        Green | Rectangular | 2
        Green | Round       | 1
        Red   | Round       | 2
        >>> marbles.groups(["Color", "Shape"], sum)
        Color | Shape       | Amount sum | Price sum
        Blue  | Rectangular | 12         | 2
        Green | Rectangular | 15         | 2.7
        Green | Round       | 2          | 1
        Red   | Round       | 11         | 3.05
        """
        collect = _zero_on_type_error(collect)
        columns = []
        labels = self._as_labels(labels)
        for label in labels:
            if label not in self.labels:
                raise ValueError("All labels must exist in the table")
            columns.append(self._get_column(label))
        grouped = self.group(list(zip(*columns)), lambda s: s)
        grouped._columns.popitem(last=False) # Discard the column of tuples

        # Flatten grouping values and move them to front
        counts = [len(v) for v in grouped[0]]
        for label in labels[::-1]:
            grouped[label] = grouped.apply(_assert_same, label)
            grouped.move_to_start(label)

        # Aggregate other values
        if collect is None:
            count = 'count' if 'count' not in labels else self._unused_label('count')
            return grouped.select(labels).with_column(count, counts)
        else:
            for label in grouped.labels:
                if label in labels:
                    continue
                column = [collect(v) for v in grouped[label]]
                del grouped[label]
                grouped[_collected_label(collect, label)] = column
            return grouped

    def pivot(self, columns, rows, values=None, collect=None, zero=None):
        """Generate a table with a column for rows (or a column for each row
        in rows list) and a column for each unique value in columns. Each row
        counts/aggregates the values that match both row and column.

        columns -- column label in self
        rows -- column label or a list of column labels
        values -- column label in self (or None to produce counts)
        collect -- aggregation function over values
        zero -- zero value for non-existent row-column combinations
        """
        if collect is not None and values is None:
            raise TypeError('collect requires values to be specified')
        if values is not None and collect is None:
            raise TypeError('values requires collect to be specified')
        rows = self._as_labels(rows)
        if values is None:
            selected = self.select([columns] + rows)
        else:
            selected = self.select([columns, values] + rows)
        grouped = selected.groups([columns] + rows, collect)

        # Generate existing combinations of values from columns in rows
        rows_values = sorted(list(set(self.select(rows).rows)))
        pivoted = type(self)(rows).with_rows(rows_values)

        # Generate other columns and add them to pivoted
        by_columns = grouped.index_by(columns)
        for label in sorted(by_columns):
            tuples = [t[1:] for t in by_columns[label]] # Discard column value
            column = _fill_with_zeros(rows_values, tuples, zero)
            pivot = self._unused_label(str(label))
            pivoted[pivot] = column
        return pivoted

    def pivot_bin(self, pivot_columns, value_column, bins=None, **vargs) :
        """Form a table with columns formed by the unique tuples in pivot_columns
        containing counts per bin of the values associated with each tuple in the value_column.

        By default, bins are chosen to contain all values in the value_column. The
        following named arguments from numpy.histogram can be applied to
        specialize bin widths:

        Args:
            ``bins`` (int or sequence of scalars): If bins is an int,
                it defines the number of equal-width bins in the given range
                (10, by default). If bins is a sequence, it defines the bin
                edges, including the rightmost edge, allowing for non-uniform
                bin widths.

            ``range`` ((float, float)): The lower and upper range of
                the bins. If not provided, range contains all values in the
                table. Values outside the range are ignored.

            ``normed`` (bool): If False, the result will contain the number of
                samples in each bin. If True, the result is normalized such that
                the integral over the range is 1.
        """
        pivot_columns = _as_labels(pivot_columns)
        selected = self.select(pivot_columns + [value_column])
        grouped = selected.groups(pivot_columns, collect=lambda x:x)

        # refine bins by taking a histogram over all the data
        if bins is not None:
            vargs['bins'] = bins
        _, rbins = np.histogram(self[value_column],**vargs)
        # create a table with these bins a first column and counts for each group
        vargs['bins'] = rbins
        binned = Table().with_column('bin',rbins)
        for group in grouped.rows:
            col_label = "-".join(map(str,group[0:-1]))
            col_vals = group[-1]
            counts,_ = np.histogram(col_vals,**vargs)
            binned[col_label] = np.append(counts,0)
        return binned

    def stack(self, key, labels=None):
        """Takes k original columns and returns two columns, with col. 1 of
        all column names and col. 2 of all associated data.
        """
        rows, labels = [], labels or self.labels
        for row in self.rows:
            [rows.append((getattr(row, key), k, v)) for k, v in row.asdict().items()
             if k != key and k in labels]
        return type(self)([key, 'column', 'value']).with_rows(rows)

    def join(self, column_label, other, other_label=None):
        """Generate a table with the columns of self and other, containing rows
        for all values of a column that appear in both tables.
        If a join value appears more than once in self, each row will be used,
        but in the other table, only the first of each will be used.

        If the result is empty, return None.
        """
        if self.num_rows == 0 or other.num_rows == 0:
            return None
        if not other_label:
            other_label = column_label

        self_rows = self.index_by(column_label)
        other_rows = other.index_by(other_label)

        # Gather joined rows from self_rows that have join values in other_rows
        joined_rows = []
        for label, rows in self_rows.items():
            if label in other_rows:
                other_row = other_rows[label][0]
                joined_rows += [row + other_row for row in rows]
        if not joined_rows:
            return None

        labels = list(self.labels)
        labels += [self._unused_label(s) for s in other.labels]
        joined = type(self)(labels).with_rows(joined_rows)
        del joined[self._unused_label(other_label)] # Remove redundant column
        return joined.move_to_start(column_label).sort(column_label)

    def stats(self, ops=(min, max, np.median, sum)):
        """Compute statistics for each column and place them in a table."""
        names = [op.__name__ for op in ops]
        ops = [_zero_on_type_error(op) for op in ops]
        columns = [[op(column) for op in ops] for column in self.columns]
        table = Table().with_columns(zip(self.labels, columns))
        stats = table._unused_label('statistic')
        table[stats] = names
        table.move_to_start(stats)
        return table

    def _as_label(self, index_or_label):
        """Convert index to label."""
        if isinstance(index_or_label, str):
            return index_or_label
        if isinstance(index_or_label, numbers.Integral):
            return self.labels[index_or_label]
        else:
            raise ValueError(str(index_or_label) + ' is not a label or index')

    def _as_labels(self, label_or_labels):
        """Convert single label to list and convert indices to labels."""
        return [self._as_label(s) for s in _as_labels(label_or_labels)]

    def _unused_label(self, label):
        """Generate an unused label."""
        original = label
        existing = self.labels
        i = 2
        while label in existing:
            label = '{}_{}'.format(original, i)
            i += 1
        return label

    def _get_column(self, column_or_label):
        """Convert label to column and check column length."""
        c = column_or_label
        if isinstance(c, collections.Hashable) and c in self.labels:
            return self[c]
        elif isinstance(c, numbers.Integral):
            return self[c]
        elif isinstance(c, str):
            raise ValueError('label "{}" not in labels {}'.format(c, self.labels))
        else:
            assert len(c) == self.num_rows, 'column length mismatch'
            return c

    def percentile(self, p):
        """Returns a new table with one row containing the pth percentile for
        each column.

        Assumes that each column only contains one type of value.

        Returns a new table with one row and the same column labels.
        The row contains the pth percentile of the original column, where the
        pth percentile of a column is the smallest value that at at least as
        large as the p% of numbers in the column.

        >>> table = Table().with_columns([
        ...     'count',  [9, 3, 3, 1],
        ...     'points', [1, 2, 2, 10]])
        >>> table
        count | points
        9     | 1
        3     | 2
        3     | 2
        1     | 10
        >>> table.percentile(80)
        count | points
        9     | 10
        """
        percentiles = [[_util.percentile(p, column)] for column in self.columns]
        return self._with_columns(percentiles)

    def sample(self, k=None, with_replacement=False, weights=None):
        """Returns a new table where k rows are randomly sampled from the
        original table.

        Kwargs:
            k (int or None): If None (default), all the rows in the table are
                sampled. If an integer, k rows from the original table are
                sampled.

            with_replacement (bool): If False (default), samples the rows
                without replacement. If True, samples the rows with replacement.

            weights (list/array or None): If None (default), samples the rows
                using a uniform random distribution. If a list/array is passed
                in, it must be the same length as the number of rows in the
                table and the values must sum to 1. The rows will then be
                sampled according the the probability distribution in
                ``weights``.

        Returns:
            A new instance of ``Table``.

        >>> jobs = Table().with_columns([
        ...     'job',  ['a', 'b', 'c', 'd'],
        ...     'wage', [10, 20, 15, 8]])
        >>> jobs
        job  | wage
        a    | 10
        b    | 20
        c    | 15
        d    | 8
        >>> jobs.sample() # doctest: +SKIP
        job  | wage
        b    | 20
        c    | 15
        a    | 10
        d    | 8
        >>> jobs.sample(k = 2) # doctest: +SKIP
        job  | wage
        b    | 20
        c    | 15
        >>> jobs.sample(k = 2, with_replacement = True,
        ...     weights = [0.5, 0.5, 0, 0]) # doctest: +SKIP
        job  | wage
        a    | 10
        a    | 10
        """
        n = self.num_rows
        if k is None:
            k = n
        index = np.random.choice(n, k, replace=with_replacement, p=weights)
        columns = [[c[i] for i in index] for c in self.columns]
        sample = self._with_columns(columns)
        return sample

    def sample_from_distribution(self, distribution, k, proportions=False):
        """Returns a new table with the same number of rows and a new column.
        The values in the distribution column are define a multinomial.
        They are replaced by sample counts/proportions in the output.

        >>> sizes = Table(['size', 'count']).with_rows([
        ...     ['small', 50],
        ...     ['medium', 100],
        ...     ['big', 50],
        ... ])
        >>> sizes.sample_from_distribution('count', 1000) # doctest: +SKIP
        size   | count | count sample
        small  | 50    | 239
        medium | 100   | 496
        big    | 50    | 265
        >>> sizes.sample_from_distribution('count', 1000, True) # doctest: +SKIP
        size   | count | count sample
        small  | 50    | 0.24
        medium | 100   | 0.51
        big    | 50    | 0.25
        """
        dist = self._get_column(distribution)
        total = sum(dist)
        assert total > 0 and np.all(dist >= 0), 'Counts or a distribution required'
        dist = dist/sum(dist)
        sample = np.random.multinomial(k, dist)
        if proportions:
            sample = sample / sum(sample)
        label = self._unused_label(self._as_label(distribution) + ' sample')
        return self.with_column(label, sample)

    def split(self, k):
        """Returns a tuple of two tables where the first table contains
        ``k`` rows randomly sampled and the second contains the remaining rows.

        Args:
            ``k`` (int): The number of rows randomly sampled into the first
                table. ``k`` must be between 1 and ``num_rows - 1``.

        Raises:
            ``ValueError``: ``k`` is not between 1 and ``num_rows - 1``.

        Returns:
            A tuple containing two instances of ``Table``.

        >>> jobs = Table().with_columns([
        ...     'job',  ['a', 'b', 'c', 'd'],
        ...     'wage', [10, 20, 15, 8]])
        >>> jobs
        job  | wage
        a    | 10
        b    | 20
        c    | 15
        d    | 8
        >>> sample, rest = jobs.split(3)
        >>> sample # doctest: +SKIP
        job  | wage
        c    | 15
        a    | 10
        b    | 20
        >>> rest # doctest: +SKIP
        job  | wage
        d    | 8
        """
        if not 1 <= k <= self.num_rows - 1:
            raise ValueError("Invalid value of k. k must be between 1 and the"
                             "number of rows - 1")

        rows = [self.rows[index] for index in
                np.random.permutation(self.num_rows)]
        cls = type(self)
        first = cls(self.labels).with_rows(rows[:k])
        rest = cls(self.labels).with_rows(rows[k:])
        for column_label in self._formats:
            first._formats[column_label] = self._formats[column_label]
            rest._formats[column_label] = self._formats[column_label]
        return first, rest

    def with_row(self, row):
        """Return a table with an additional row.

        Args:
            ``row`` (sequence): A value for each column.

        Raises:
            ``ValueError``: If the row length differs from the column count.

        >>> tiles = Table(['letter', 'count', 'points'])
        >>> tiles.with_row(['c', 2, 3]).with_row(['d', 4, 2])
        letter | count | points
        c      | 2     | 3
        d      | 4     | 2
        """
        self = self.copy()
        self.append(row)
        return self

    def with_rows(self, rows):
        """Return a table with additional rows.

        Args:
            ``rows`` (sequence of sequences): Each row has a value per column.

            If ``rows`` is a 2-d array, its shape must be (_, n) for n columns.

        Raises:
            ``ValueError``: If a row length differs from the column count.

        >>> tiles = Table(['letter', 'count', 'points'])
        >>> tiles.with_rows([['c', 2, 3], ['d', 4, 2]])
        letter | count | points
        c      | 2     | 3
        d      | 4     | 2
        """
        self = self.copy()
        self.append(self._with_columns(zip(*rows)))
        return self

    def with_column(self, label, values):
        """Return a table with an additional or replaced column.

        Args:
            ``label`` (str): The column label. If an existing label is used,
                that column will be replaced in the returned table.

            ``values`` (single value or sequence): If a single value, every
                value in the new column is ``values``.

                If a sequence, the new column contains the values in
                ``values``. ``values`` must be the same length as the table.

        Raises:
            ``ValueError``: If
                - ``label`` is not a valid column name
                - ``values`` is a list/array and does not have the same length
                  as the number of rows in the table.

        >>> tiles = Table().with_columns([
        ...     'letter', ['c', 'd'],
        ...     'count',  [2, 4],
        ... ])
        >>> tiles.with_column('points', [3, 2])
        letter | count | points
        c      | 2     | 3
        d      | 4     | 2
        >>> tiles.with_column('count', 1)
        letter | count
        c      | 1
        d      | 1
        """
        new_table = self.copy()
        new_table.append_column(label, values)
        return new_table

    def with_columns(self, labels_and_values):
        """Return a table with additional or replaced columns.

        Args:
            ``labels_and_values``: An alternating list of labels and values or
                a list of label-values pairs.

        >>> Table().with_columns([
        ...     'letter', ['c', 'd'],
        ...     'count',  [2, 4],
        ... ])
        letter | count
        c      | 2
        d      | 4
        >>> Table().with_columns([
        ...     ['letter', ['c', 'd']],
        ...     ['count',  [2, 4]],
        ... ])
        letter | count
        c      | 2
        d      | 4
        >>> Table().with_columns({'letter': ['c', 'd']})
        letter
        c
        d
        """
        if isinstance(labels_and_values, collections.abc.Mapping):
            labels_and_values = list(labels_and_values.items())
        if not isinstance(labels_and_values, collections.abc.Sequence):
            labels_and_values = list(labels_and_values)
        if not labels_and_values:
            return self
        first = labels_and_values[0]
        if not isinstance(first, str) and hasattr(first, '__iter__'):
            for pair in labels_and_values:
                assert len(pair) == 2, 'incorrect columns format'
            labels_and_values = [x for pair in labels_and_values for x in pair]
        assert len(labels_and_values) % 2 == 0, 'Even length sequence required'
        for i in range(0, len(labels_and_values), 2):
            label, values = labels_and_values[i], labels_and_values[i+1]
            self = self.with_column(label, values)
        return self

    def relabeled(self, label, new_label):
        """Returns a table with label changed to new_label.

        ``label`` and ``new_label`` may be single values or lists
        specifying column labels to be changed and their new corresponding
        labels.

        Args:
            ``label`` (str or sequence of str): The label(s) of
                columns to be changed.

            ``new_label`` (str or sequence of str): The new label(s) of
                columns to be changed. Same number of elements as label.

        >>> tiles = Table(['letter', 'count'])
        >>> tiles = tiles.with_rows([['c', 2], ['d', 4]])
        >>> tiles.relabeled('count', 'number')
        letter | number
        c      | 2
        d      | 4
        """
        copy = self.copy()
        copy.relabel(label, new_label)
        return copy

    # Deprecated
    def with_relabeling(self, *args):
        warnings.warn("with_relabeling is deprecated; use relabeled", FutureWarning)
        return self.relabeled(*args)

    def bin(self, select=None, **vargs):
        """Group values by bin and compute counts per bin by column.

        By default, bins are chosen to contain all values in all columns. The
        following named arguments from numpy.histogram can be applied to
        specialize bin widths:

        If the original table has n columns, the resulting binned table has
        n+1 columns, where column 0 contains the lower bound of each bin.

        Args:
            ``select`` (columns): Columns to be binned. If None, all columns
                are binned.

            ``bins`` (int or sequence of scalars): If bins is an int,
                it defines the number of equal-width bins in the given range
                (10, by default). If bins is a sequence, it defines the bin
                edges, including the rightmost edge, allowing for non-uniform
                bin widths.

            ``range`` ((float, float)): The lower and upper range of
                the bins. If not provided, range contains all values in the
                table. Values outside the range are ignored.

            ``density`` (bool): If False, the result will contain the number of
                samples in each bin. If True, the result is the value of the
                probability density function at the bin, normalized such that
                the integral over the range is 1. Note that the sum of the
                histogram values will not be equal to 1 unless bins of unity
                width are chosen; it is not a probability mass function.
        """
        if select is not None:
            self = self.select(select)
        if 'normed' in vargs:
            vargs.setdefault('density', vargs.pop('normed'))
        density = vargs.get('density', False)
        tag = 'density' if density else 'count'

        cols = list(self._columns.values())
        _, bins = np.histogram(cols, **vargs)

        binned = Table().with_column('bin', bins)
        for label in self.labels:
            counts, _ = np.histogram(self[label], bins=bins, density=density)
            binned[label + ' ' + tag] = np.append(counts, 0)
        return binned

    ##########################
    # Exporting / Displaying #
    ##########################

    def __str__(self):
        return self.as_text(self.max_str_rows)

    __repr__ = __str__

    def _repr_html_(self):
        return self.as_html(self.max_str_rows)

    def show(self, max_rows=0):
        """Display the table."""
        IPython.display.display(IPython.display.HTML(self.as_html(max_rows)))

    max_str_rows = 10

    def as_text(self, max_rows=0, sep=" | "):
        """Format table as text."""
        if not max_rows or max_rows > self.num_rows:
            max_rows = self.num_rows
        omitted = max(0, self.num_rows - max_rows)
        labels = self._columns.keys()
        fmts = [self._formats.get(k, self.formatter.format_column(k, v[:max_rows])) for
            k, v in self._columns.items()]
        rows = [[fmt(label, label=True) for fmt, label in zip(fmts, labels)]]
        for row in itertools.islice(self.rows, max_rows):
            rows.append([f(v, label=False) for v, f in zip(row, fmts)])
        lines = [sep.join(row) for row in rows]
        if omitted:
            lines.append('... ({} rows omitted)'.format(omitted))
        return '\n'.join([line.rstrip() for line in lines])

    def as_html(self, max_rows=0):
        """Format table as HTML."""
        if not max_rows or max_rows > self.num_rows:
            max_rows = self.num_rows
        omitted = max(0, self.num_rows - max_rows)
        labels = self.labels
        lines = [
            (0, '<table border="1" class="dataframe">'),
            (1, '<thead>'),
            (2, '<tr>'),
            (3, ' '.join('<th>' + label + '</th>' for label in labels)),
            (2, '</tr>'),
            (1, '</thead>'),
            (1, '<tbody>'),
        ]
        fmts = [self._formats.get(k, self.formatter.format_column(k, v[:max_rows])) for
            k, v in self._columns.items()]
        fmts = [(lambda f: lambda v, label=False: v.as_html() if hasattr(v, 'as_html') else f(v))(f) for f in fmts]
        for row in itertools.islice(self.rows, max_rows):
            lines += [
                (2, '<tr>'),
                (3, ' '.join('<td>' + fmt(v, label=False) + '</td>' for
                    v, fmt in zip(row, fmts))),
                (2, '</tr>'),
                (1, '</tbody>'),
            ]
        lines.append((0, '</table>'))
        if omitted:
            lines.append((0, '<p>... ({} rows omitted)</p'.format(omitted)))
        return '\n'.join(4 * indent * ' ' + text for indent, text in lines)

    def index_by(self, column_or_label):
        """Return a dict keyed by values in a column that contains lists of
        rows corresponding to each value.
        """
        column = self._get_column(column_or_label)
        index = {}
        for key, row in zip(column, self.rows):
            index.setdefault(key, []).append(row)
        return index

    def to_df(self):
        """Convert the table to a Pandas DataFrame."""
        return pandas.DataFrame(self._columns)

    def to_csv(self, filename):
        """Creates a CSV file with the provided filename.

        The CSV is created in such a way that if we run
        ``table.to_csv('my_table.csv')`` we can recreate the same table with
        ``Table.read_table('my_table.csv')``.

        Args:
            ``filename`` (str): The filename of the output CSV file.

        Returns:
            None, outputs a file with name ``filename``.

        >>> jobs = Table().with_columns([
        ...     'job',  ['a', 'b', 'c', 'd'],
        ...     'wage', [10, 20, 15, 8]])
        >>> jobs
        job  | wage
        a    | 10
        b    | 20
        c    | 15
        d    | 8
        >>> jobs.to_csv('my_table.csv') # doctest: +SKIP
        <outputs a file called my_table.csv in the current directory>
        """
        # index=False avoids row numbers in the output
        self.to_df().to_csv(filename, index=False)

    def to_array(self):
        """Convert the table to a NumPy array."""
        dt = np.dtype(list(zip(self.labels, (c.dtype for c in self.columns))))
        arr = np.empty_like(self.columns[0], dt)
        for label in self.labels:
            arr[label] = self[label]
        return arr

    ##################
    # Visualizations #
    ##################

    # As RGB tuples
    chart_colors = (
        (0.0, 0.102, 0.267),
        (1.0, 0.784, 0.0),
        (0.341, 0.408, 0.518),
        (0.702, 0.612, 0.302),
        (0.463, 0.537, 0.282),
        (0.024, 0.482, 0.761),
        (0.984, 0.314, 0.071),
        (0.098, 0.22, 0.122),
        (0.298, 0.235, 0.216),
    )

    default_hist_alpha = 0.7

    default_options = {
        'alpha': 0.7,
    }

    def plot(self, column_for_xticks=None, select=None, overlay=True, **vargs):
        """Plot line charts for the table.

        Each plot is labeled using the values in `column_for_xticks` and one
        plot is produced for every other column (or for the columns designated
        by `select`).

        Every selected column except for `column_for_xticks` must be numerical.

        Args:
            column_for_xticks (str/array): A column containing x-axis labels

        Kwargs:
            overlay (bool): create a chart with one color per data column;
                if False, each will be displayed separately.

            vargs: Additional arguments that get passed into `plt.plot`.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
                for additional arguments that can be passed into vargs.
        """
        options = self.default_options.copy()
        options.update(vargs)

        if column_for_xticks is not None:
            x_data, y_labels = self._split_column_and_labels(column_for_xticks)
            x_label = self._as_label(column_for_xticks)
        else:
            x_data, y_labels = None, self.labels
            x_label = None
        if select is not None:
            y_labels = self._as_labels(select)

        if x_data is not None:
            self = self.sort(x_data)
            x_data = np.sort(x_data)

        def draw(axis, label, color):
            if x_data is None:
                axis.plot(self[label], color=color, **options)
            else:
                axis.plot(x_data, self[label], color=color, **options)

        self._visualize(x_label, y_labels, None, overlay, draw, _vertical_x)

    def bar(self, column_for_categories=None, select=None, overlay=True, **vargs):
        """Plot bar charts for the table.

        Each plot is labeled using the values in `column_for_categories` and
        one plot is produced for every other column (or for the columns
        designated by `select`).

        Every selected except column for `column_for_categories` must be numerical.

        Args:
            column_for_categories (str): A column containing x-axis categories

        Kwargs:
            overlay (bool): create a chart with one color per data column;
                if False, each will be displayed separately.

            vargs: Additional arguments that get passed into `plt.bar`.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.bar
                for additional arguments that can be passed into vargs.
        """
        options = self.default_options.copy()
        options.update(vargs)

        xticks, labels = self._split_column_and_labels(column_for_categories)
        if select is not None:
            labels = self._as_labels(select)

        index = np.arange(self.num_rows)

        def draw(axis, label, color):
            axis.bar(index-0.5, self[label], 1.0, color=color, **options)

        def annotate(axis, ticks):
            if (ticks is not None) :
                tick_labels = [ticks[int(l)] if 0<=l<len(ticks) else '' for l in axis.get_xticks()]
                axis.set_xticklabels(tick_labels, stretch='ultra-condensed')

        self._visualize(column_for_categories, labels, xticks, overlay, draw, annotate)

    def barh(self, column_for_categories=None, select=None, overlay=True, **vargs):
        """Plot horizontal bar charts for the table.

        Each plot is labeled using the values in `column_for_categories` and
        one plot is produced for every other column (or for the columns
        designated by `select`).

        Every selected except column for `column_for_categories` must be numerical.

        Args:
            column_for_categories (str): A column containing y-axis categories

        Kwargs:
            overlay (bool): create a chart with one color per data column;
                if False, each will be displayed separately.

            vargs: Additional arguments that get passed into `plt.barh`.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.barh
                for additional arguments that can be passed into vargs.

        >>> t = Table().with_columns([
        ...     'Furniture', ['chairs', 'tables', 'desks'],
        ...     'Count', [6, 1, 2],
        ...     'Price', [10, 20, 30]
        ...     ])
        >>> t
        Furniture | Count | Price
        chairs    | 6     | 10
        tables    | 1     | 20
        desks     | 2     | 30
        >>> furniture_table.barh('Furniture') # doctest: +SKIP
        <bar graph with furniture as categories and bars for count and price>
        >>> furniture_table.barh('Furniture', 'Price') # doctest: +SKIP
        <bar graph with furniture as categories and bars for price>
        >>> furniture_table.barh('Furniture', [1, 2]) # doctest: +SKIP
        <bar graph with furniture as categories and bars for count and price>
        """
        options = self.default_options.copy()
        options.update(vargs)

        yticks, labels = self._split_column_and_labels(column_for_categories)
        if select is not None:
            labels = self._as_labels(select)
        n = len(labels)

        index = np.arange(self.num_rows)
        margin = 0.1
        width = 1 - 2 * margin
        if overlay:
            width /= len(labels)

        def draw(axis, label, color):
            if overlay:
                ypos = index + margin + (1-2*margin)*(n - 1 - labels.index(label))/n
            else:
                ypos = index
            # barh plots entries in reverse order from bottom to top
            axis.barh(ypos, self[label][::-1], width,  color=color, **options)

        def annotate(axis, ticks):
            axis.set_yticks(index+0.5) # Center labels on bars
            # barh plots entries in reverse order from bottom to top
            axis.set_yticklabels(ticks[::-1], stretch='ultra-condensed')
            axis.set_xlabel(axis.get_ylabel())
            if isinstance(column_for_categories, str):
                axis.set_ylabel(column_for_categories)

        height = max(4, len(index)/2)
        if 'height' in vargs:
            height = vargs.pop('height')

        self._visualize('', labels, yticks, overlay, draw, annotate, height=height)

    def scatter(self, column_for_x, select=None, overlay=True, fit_line=False,
        colors=None, labels=None, **vargs):
        """Creates scatterplots, optionally adding a line of best fit.

        Each plot uses the values in `column_for_x` for horizontal positions.
        One plot is produced for every other column as y (or for the columns
        designated by `select`).

        Every selected except column for `column_for_categories` must be numerical.

        Args:
            ``column_for_x`` (str): The name to use for the x-axis values of the
                scatter plots.

        Kwargs:
            ``overlay`` (bool): create a chart with one color per data column;
                if False, each will be displayed separately.

            ``fit_line`` (bool): draw a line of best fit for each set of points

            ``vargs``: Additional arguments that get passed into `plt.scatter`.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter
                for additional arguments that can be passed into vargs. These
                include: `marker` and `norm`, to name a couple.

            ``colors``: A column of colors (labels or numeric values)

            ``labels``: A column of text labels to annotate dots

        >>> table = Table().with_columns([
        ...     'x', [9, 3, 3, 1],
        ...     'y', [1, 2, 2, 10],
        ...     'z', [3, 4, 5, 6]])
        >>> table
        x    | y    | z
        9    | 1    | 3
        3    | 2    | 4
        3    | 2    | 5
        1    | 10   | 6
        >>> table.scatter('x') # doctest: +SKIP
        <scatterplot of values in y and z on x>

        >>> table.scatter('x', overlay=False) # doctest: +SKIP
        <scatterplot of values in y on x>
        <scatterplot of values in z on x>

        >>> table.scatter('x', fit_line=True) # doctest: +SKIP
        <scatterplot of values in y and z on x with lines of best fit>
        """
        options = self.default_options.copy()
        options.update(vargs)

        x_data, y_labels =  self._split_column_and_labels(column_for_x)
        if select is not None:
            y_labels = self._as_labels(select)

        def draw(axis, label, color):
            if colors is not None:
                color = self[colors]
            elif 'color' in options:
                color = options.pop('color')
            y_data = self[label]
            axis.scatter(x_data, y_data, color=color, **options)
            if fit_line:
                m, b = np.polyfit(x_data, self[label], 1)
                minx, maxx = np.min(x_data),np.max(x_data)
                axis.plot([minx,maxx],[m*minx+b,m*maxx+b], color=color)
            if labels is not None:
                for x, y, label in zip(x_data, y_data, self[labels]):
                    axis.annotate(label, (x, y),
                        xytext=(-20, 20),
                        textcoords='offset points', ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
                        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', color='black'))

        x_label = self._as_label(column_for_x)
        self._visualize(x_label, y_labels, None, overlay, draw, _vertical_x, width=5, height=5)

    def _visualize(self, x_label, y_labels, ticks, overlay, draw, annotate, width=6, height=4):
        """Generic visualization that overlays or separates the draw function.

        Raises:
            ValueError: The Table contains non-numerical values in columns
            other than `column_for_categories`
        """
        for label in y_labels:
            if not all(isinstance(x, numbers.Real) for x in self[label]):
                raise ValueError("The column '{0}' contains non-numerical "
                    "values. A plot cannot be drawn for this column."
                    .format(label))

        n = len(y_labels)
        colors = list(itertools.islice(itertools.cycle(self.chart_colors), n))
        if overlay and n > 1:
            _, axis = plt.subplots(figsize=(width, height))
            if x_label is not None:
                axis.set_xlabel(x_label)
            for label, color in zip(y_labels, colors):
                draw(axis, label, color)
            if ticks is not None:
                annotate(axis, ticks)
            axis.legend(y_labels, loc=2, bbox_to_anchor=(1.05, 1))
            Table.plots.append(axis)
        else:
            fig, axes = plt.subplots(n, 1, figsize=(width, height*n))
            if not isinstance(axes, collections.Iterable):
                axes=[axes]
            for axis, y_label, color in zip(axes, y_labels, colors):
                draw(axis, y_label, color)
                axis.set_ylabel(y_label, fontsize=16)
                if x_label is not None:
                    axis.set_xlabel(x_label, fontsize=16)
                if ticks is not None:
                    annotate(axis, ticks)
                Table.plots.append(axis)

    def _split_column_and_labels(self, column_or_label):
        """Return the specified column and labels of other columns."""
        column = None if column_or_label is None else self._get_column(column_or_label)
        labels = [label for i, label in enumerate(self.labels) if column_or_label not in (i, label)]
        return column, labels

    def pivot_hist(self, pivot_column_label, value_column_label, overlay=True, **vargs):
        """Draw histograms of each category in a column."""
        pvt_labels = np.unique(self[pivot_column_label])
        pvt_columns = [self[value_column_label][np.where(self[pivot_column_label] == pivot)] for pivot in pvt_labels]
        n = len(pvt_labels)
        colors = list(itertools.islice(itertools.cycle(self.chart_colors), n))
        if overlay:
            plt.figure(figsize=(6, 4))
            vals, bins, patches = plt.hist(pvt_columns, color=colors, **vargs)
            plt.legend(pvt_labels)
        else:
            _, axes = plt.subplots(n, 1, figsize=(6, 4 * n))
            vals = []
            bins = None
            for axis, label, column, color in zip(axes, pvt_labels, pvt_columns, colors):
                if isinstance(bins, np.ndarray):
                    avals, abins, patches = axis.hist(column, color=color, bins=bins, **vargs)
                else:
                    avals, abins, patches = axis.hist(column, color=color, **vargs)
                axis.set_xlabel(label, fontsize=16)
                vals.append(avals)
                if not isinstance(bins, np.ndarray):
                    bins = abins
                else:
                    assert bins.all() == abins.all(), "Inconsistent bins in hist"
        t = Table()
        t['start'] = bins[0:-1]
        t['end'] = bins[1:]
        for label, column in zip(pvt_labels,vals):
            t[label] = column

    def hist(self, select=None, overlay=True, bins=None, counts=None, unit=None, **vargs):
        """Plots one histogram for each column in the table.

        Every column must be numerical.

        Kwargs:
            overlay (bool): If True, plots 1 chart with all the histograms
                overlaid on top of each other (instead of the default behavior
                of one histogram for each column in the table). Also adds a
                legend that matches each bar color to its column.

            bins (column name or list): Lower bound for each bin in the
                histogram. If None, bins will be chosen automatically.

            counts (column name or column): A column of counted values.
                All other columns are treated as counts of these values.
                If None, each value in each row is assigned a count of 1.

            vargs: Additional arguments that get passed into :func:plt.hist.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hist
                for additional arguments that can be passed into vargs. These
                include: `range`, `normed`, `cumulative`, and `orientation`,
                to name a few.

        >>> t = Table().with_columns([
        ...     'count',  [9, 3, 3, 1],
        ...     'points', [1, 2, 2, 10]])
        >>> t
        count | points
        9     | 1
        3     | 2
        3     | 2
        1     | 10
        >>> t.hist() # doctest: +SKIP
        <histogram of values in count>
        <histogram of values in points>

        >>> t = Table().with_columns([
        ...     'value',      [101, 102, 103],
        ...     'proportion', [0.25, 0.5, 0.25]])
        >>> t.hist(counts='value') # doctest: +SKIP
        <histogram of values in prop weighted by corresponding values in value>
        """
        if select is not None:
            self = self.select(select)

        # Check for non-numerical values and raise a ValueError if any found
        for col in self:
            if any(isinstance(cell, np.flexible) for cell in self[col]):
                raise ValueError("The column '{0}' contains non-numerical "
                    "values. A histogram cannot be drawn for this table."
                    .format(col))

        columns = self._columns.copy()

        if bins is not None:
            if isinstance(bins, collections.Hashable) and bins in self.labels:
                bins = np.unique(self[bins])
            vargs['bins'] = bins

        if 'normed' not in vargs:
            vargs['normed'] = True
        percentage = plt.FuncFormatter(lambda x, _: "{:g}".format(100*x))

        counted_values = counted_label = None
        if counts is not None:
            counted_values = self._get_column(counts)
            counted_label = 'counts'
            if isinstance(counts, str) and counts in self.labels:
                columns.pop(counts)
                counted_label = counts

        n = len(columns)
        colors = [rgb_color + (self.default_hist_alpha,) for rgb_color in
            itertools.islice(itertools.cycle(self.chart_colors), n)]
        if overlay and n > 1:
            # Reverse because legend prints bottom-to-top
            column_keys = list(columns.keys())[::-1]
            values = list(columns.values())[::-1]
            colors = list(colors)[::-1]
            if counted_values is not None:
                vargs['weights'] = np.transpose(values)
                values = np.repeat(counted_values, n).reshape(-1,n)
            vargs.setdefault('histtype', 'stepfilled')
            figure = plt.figure(figsize=(6, 4))
            print(values, colors, vargs)
            plt.hist(values, color=colors, **vargs)
            axis = figure.get_axes()[0]
            _vertical_x(axis)
            if vargs['normed']:
                axis.set_ylabel('Percent per ' + (unit if unit else 'unit'))
                axis.yaxis.set_major_formatter(percentage)
            else:
                axis.set_ylabel('Count')
            if unit:
                axis.set_xlabel('(' + unit + ')', fontsize=16)
            plt.legend(columns.keys(), loc=2, bbox_to_anchor=(1.05, 1))
            Table.plots.append(axis)
        else:
            _, axes = plt.subplots(n, 1, figsize=(6, 4 * n))
            # Use stepfilled when there are too many bins
            if isinstance(bins, numbers.Integral) and bins > 76 or hasattr(bins, '__len__') and len(bins) > 76:
                vargs.setdefault('histtype', 'stepfilled')
            if n == 1:
                axes = [axes]
            for axis, label, color in zip(axes, columns.keys(), colors):
                if vargs['normed']:
                    axis.set_ylabel('Percent per ' + (unit if unit else 'unit'))
                    axis.yaxis.set_major_formatter(percentage)
                else:
                    axis.set_ylabel('Count')
                x_unit = ' (' + unit + ')' if unit else ''
                if counted_values is None:
                    values = columns[label]
                    axis.set_xlabel(label + x_unit, fontsize=16)
                else:
                    values = counted_values
                    axis.set_xlabel(counted_label + x_unit, fontsize=16)
                    vargs['weights'] = columns[label]
                axis.hist(values, color=color, **vargs)
                _vertical_x(axis)
                Table.plots.append(axis)

    def boxplot(self, **vargs):
        """Plots a boxplot for the table.

        Every column must be numerical.

        Kwargs:
            vargs: Additional arguments that get passed into `plt.boxplot`.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.boxplot
                for additional arguments that can be passed into vargs. These include
                `vert` and `showmeans`.

        Returns:
            None

        Raises:
            ValueError: The Table contains columns with non-numerical values.

        >>> table = Table().with_columns([
        ...     'test1', [92.5, 88, 72, 71, 99, 100, 95, 83, 94, 93],
        ...     'test2', [89, 84, 74, 66, 92, 99, 88, 81, 95, 94]])
        >>> table
        test1 | test2
        92.5  | 89
        88    | 84
        72    | 74
        71    | 66
        99    | 92
        100   | 99
        95    | 88
        83    | 81
        94    | 95
        93    | 94
        >>> table.boxplot() # doctest: +SKIP
        <boxplot of test1 and boxplot of test2 side-by-side on the same figure>
        """
        # Check for non-numerical values and raise a ValueError if any found
        for col in self:
            if any(isinstance(cell, np.flexible) for cell in self[col]):
                raise ValueError("The column '{0}' contains non-numerical "
                    "values. A histogram cannot be drawn for this table."
                    .format(col))

        columns = self._columns.copy()
        vargs['labels'] = columns.keys()
        values = list(columns.values())
        plt.boxplot(values, **vargs)

    # Deprecated
    def points(self, column__lat, column__long, labels=None, colors=None, **kwargs) :
        """Draw points from latitude and longitude columns. [Deprecated]"""
        warnings.warn("points is deprecated. Use Circle.map", FutureWarning)
        latitudes = self._get_column(column__lat)
        longitudes = self._get_column(column__long)
        if labels is not None : labels = self._get_column(labels)
        if colors is not None : colors = self._get_column(colors)
        return _maps.Circle.map(latitudes, longitudes, labels=labels, colors=colors, **kwargs)


    ###########
    # Support #
    ###########

    class Row(tuple):
        _table = None  # Set by subclasses in Rows

        def __getattr__(self, column_label):
            return self[self._table.column_index(column_label)]

        def item(self, index_or_label):
            """Return the item at an index or label."""
            if isinstance(index_or_label, numbers.Integral):
                index = index_or_label
            else:
                index = self._table.column_index(index_or_label)
            return self[index]

        def __repr__(self):
            return 'Row({})'.format(', '.join('{}={}'.format(
                self._table.labels[i], v.__repr__()) for i, v in enumerate(self)))

        def asdict(self):
            return collections.OrderedDict(zip(self._table.labels, self))

    class Rows(collections.abc.Sequence):
        """An iterable view over the rows in a table."""
        def __init__(self, table):
            self._table = table
            self._labels = None

        def __getitem__(self, i):
            if isinstance(i, slice):
                return (self[j] for j in range(*i.indices(len(self))))

            labels = tuple(self._table.labels)
            if labels != self._labels:
                self._labels = labels
                self._row = type('Row', (Table.Row, ), dict(_table=self._table))
            return self._row(c[i] for c in self._table._columns.values())

        def __len__(self):
            return self._table.num_rows

        def __repr__(self):
            return '{0}({1})'.format(type(self).__name__, repr(self._table))


def _zero_on_type_error(column_fn):
    """Wrap a function on an np.ndarray to return 0 on a type error."""
    if not column_fn:
        return column_fn
    if not callable(column_fn):
        raise TypeError('column functions must be callable')
    @functools.wraps(column_fn)
    def wrapped(column):
        try:
            return column_fn(column)
        except TypeError:
            if isinstance(column, np.ndarray):
                return column.dtype.type() # A typed zero value
            else:
                raise
    return wrapped


def _fill_with_zeros(partials, rows, zero=None):
    """Find and return values from rows for all partials. In cases where no
    row matches a partial, zero is assumed as value. For a row, the first
    (n-1) fields are assumed to be the partial, and the last field,
    the value, where n is the total number of fields in each row. It is
    assumed that there is a unique row for each partial.
    partials -- single field values or tuples of field values
    rows -- table rows
    zero -- value used when no rows match a particular partial
    """
    assert len(rows) > 0
    if not _is_non_string_iterable(partials):
        # Convert partials to tuple for comparison against row slice later
        partials = [(partial,) for partial in partials]

    # Construct mapping of partials to values in rows
    mapping = {}
    for row in rows:
        mapping[tuple(row[:-1])] = row[-1]

    if zero is None:
        # Try to infer zero from given row values.
        array = np.array(tuple(mapping.values()))
        if len(array.shape) == 1:
            zero = array.dtype.type()
    return np.array([mapping.get(partial, zero) for partial in partials])


def _as_labels(column_label_or_labels):
    """Return a list of labels for a label or labels."""
    if not _is_non_string_iterable(column_label_or_labels):
        return [column_label_or_labels]
    else:
        return column_label_or_labels

def _assert_same(values):
    """Assert that all values are identical and return the unique value."""
    assert len(values) > 0
    first, rest = values[0], values[1:]
    for v in rest:
        assert v == first
    return first


def _collected_label(collect, label):
    """Label of a collected column."""
    if not collect.__name__.startswith('<'):
        return label + ' ' + collect.__name__
    else:
        return label


def _is_non_string_iterable(value):
    """Whether a value is iterable."""
    if isinstance(value, str):
        return False
    if hasattr(value, '__iter__'):
        return True
    if isinstance(value, collections.abc.Sequence):
        return True
    return False

def _vertical_x(axis, ticks=None, max_width=5):
    """Switch labels to vertical if they are long."""
    if ticks is None:
        ticks = axis.get_xticks()
    if (np.array(ticks) == np.rint(ticks)).all():
        ticks = np.rint(ticks).astype(np.int)
    if max([len(str(tick)) for tick in ticks]) > max_width:
        axis.set_xticklabels(ticks, rotation='vertical')

###################
# Slicing support #
###################

class _RowSelector(metaclass=abc.ABCMeta):
    def __init__(self, table):
        self._table = table

    def __call__(self, row_numbers_or_slice):
        return self[row_numbers_or_slice]

    @abc.abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError()


class _RowTaker(_RowSelector):
    def __getitem__(self, row_indices_or_slice):
        """Return a new Table of a sequence of rows taken by number.

        Args:
            ``row_indices_or_slice`` (integer or list of integers or slice):
            The row index, list of row indices or a slice of row indices to
            be selected.

        Returns:
            A new instance of ``Table``.

        >>> t = Table().with_columns([
        ...     'letter grade', ['A+', 'A', 'A-', 'B+', 'B', 'B-'],
        ...     'gpa', [4, 4, 3.7, 3.3, 3, 2.7]])
        >>> t
        letter grade | gpa
        A+           | 4
        A            | 4
        A-           | 3.7
        B+           | 3.3
        B            | 3
        B-           | 2.7
        >>> t.take(0)
        letter grade | gpa
        A+           | 4
        >>> t.take(5)
        letter grade | gpa
        B-           | 2.7
        >>> t.take(-1)
        letter grade | gpa
        B-           | 2.7
        >>> t.take([2, 1, 0])
        letter grade | gpa
        A-           | 3.7
        A            | 4
        A+           | 4
        >>> t.take([1, 5])
        letter grade | gpa
        A            | 4
        B-           | 2.7
        >>> t.take(range(3))
        letter grade | gpa
        A+           | 4
        A            | 4
        A-           | 3.7

        Note that ``take`` also supports NumPy-like indexing and slicing:

        >>> t.take[:3]
        letter grade | gpa
        A+           | 4
        A            | 4
        A-           | 3.7

        >>> t.take[2, 1, 0]
        letter grade | gpa
        A-           | 3.7
        A            | 4
        A+           | 4

        """
        if isinstance(row_indices_or_slice, collections.Iterable):
            columns = [np.take(column, row_indices_or_slice, axis=0)
                       for column in self._table._columns.values()]
            return self._table._with_columns(columns)

        rows = self._table.rows[row_indices_or_slice]
        if isinstance(rows, Table.Row):
            rows = [rows]
        return self._table._with_columns(zip(*rows))


class _RowExcluder(_RowSelector):
    def __getitem__(self, row_indices_or_slice):
        """Return a new Table without a sequence of rows excluded by number.

        Args:
            ``row_indices_or_slice`` (integer or list of integers or slice):
                The row index, list of row indices or a slice of row indices
                to be excluded.

        Returns:
            A new instance of ``Table``.

        >>> t = Table().with_columns([
        ...     'letter grade', ['A+', 'A', 'A-', 'B+', 'B', 'B-'],
        ...     'gpa', [4, 4, 3.7, 3.3, 3, 2.7]])
        >>> t
        letter grade | gpa
        A+           | 4
        A            | 4
        A-           | 3.7
        B+           | 3.3
        B            | 3
        B-           | 2.7
        >>> t.exclude(4)
        letter grade | gpa
        A+           | 4
        A            | 4
        A-           | 3.7
        B+           | 3.3
        B-           | 2.7
        >>> t.exclude(-1)
        letter grade | gpa
        A+           | 4
        A            | 4
        A-           | 3.7
        B+           | 3.3
        B            | 3
        >>> t.exclude([1, 3, 4])
        letter grade | gpa
        A+           | 4
        A-           | 3.7
        B-           | 2.7
        >>> t.exclude(range(3))
        letter grade | gpa
        B+           | 3.3
        B            | 3
        B-           | 2.7

        Note that ``exclude`` also supports NumPy-like indexing and slicing:

        >>> t.exclude[:3]
        letter grade | gpa
        B+           | 3.3
        B            | 3
        B-           | 2.7

        >>> t.exclude[1, 3, 4]
        letter grade | gpa
        A+           | 4
        A-           | 3.7
        B-           | 2.7
        """
        if isinstance(row_indices_or_slice, collections.Iterable):
            without_row_indices = set(row_indices_or_slice)
            rows = [row for index, row in enumerate(self._table.rows[:])
                    if index not in without_row_indices]
            return self._table._with_columns(zip(*rows))

        row_slice = row_indices_or_slice
        if not isinstance(row_slice, slice):
            row_slice %= self._table.num_rows
            row_slice = slice(row_slice, row_slice+1)
        rows = itertools.chain(self._table.rows[:row_slice.start or 0],
                               self._table.rows[row_slice.stop:])
        return self._table._with_columns(zip(*rows))

# For Sphinx: grab the docstrings from `Taker.__getitem__` and `Withouter.__getitem__`
Table.take.__doc__ = _RowTaker.__getitem__.__doc__
Table.exclude.__doc__ = _RowExcluder.__getitem__.__doc__
