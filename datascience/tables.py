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

import datascience.formats as _formats
import datascience.util as _util
from datascience.util import make_array
import datascience.predicates as _predicates

class Table(collections.abc.MutableMapping):
    """A sequence of string-labeled columns."""
    plots = collections.deque(maxlen=10)

    def __init__(self, labels=None, formatter=_formats.default_formatter):
        """Create an empty table with column labels.

        >>> tiles = Table(make_array('letter', 'count', 'points'))
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
        """Creates an empty table. Column labels are optional. [Deprecated]

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
        df = pandas.read_csv(filepath_or_buffer, *args, **vargs)
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
        for label in labels:
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
        return self.column(index_or_label)

    def __setitem__(self, index_or_label, values):
        self.append_column(index_or_label, values)

    def __delitem__(self, index_or_label):
        label = self._as_label(index_or_label)
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
        """Return a tuple of columns, each with the values in that column."""
        return tuple(self._columns.values())

    def column(self, index_or_label):
        """Return the values of a column as an array.

        table.column(label) is equivalent to table[label].

        >>> tiles = Table().with_columns(
        ...     'letter', make_array('c', 'd'),
        ...     'count',  make_array(2, 4),
        ... )

        >>> list(tiles.column('letter'))
        ['c', 'd']
        >>> tiles.column(1)
        array([2, 4])

        Args:
            label (int or str): The index or label of a column

        Returns:
            An instance of ``numpy.array``.

        Raises:
            ``ValueError``: When the ``index_or_label`` is not in the table.
        """
        if (isinstance(index_or_label, str)
                and index_or_label not in self.labels):
            raise ValueError(
                'The column "{}" is not in the table. The table contains '
                'these columns: {}'
                .format(index_or_label, ', '.join(self.labels))
            )
        if (isinstance(index_or_label, int)
                and not 0 <= index_or_label < len(self.labels)):
            raise ValueError(
                'The index {} is not in the table. Only indices between '
                '0 and {} are valid'
                .format(index_or_label, len(self.labels) - 1)
            )

        return self._columns[self._as_label(index_or_label)]

    @property
    def values(self):
        """Return data in `self` as a numpy array.

        If all columns are the same dtype, the resulting array
        will have this dtype. If there are >1 dtypes in columns,
        then the resulting array will have dtype `object`.
        """
        dtypes = [col.dtype for col in self.columns]
        if len(set(dtypes)) > 1:
            dtype = object
        else:
            dtype = None
        return np.array(self.columns, dtype=dtype).T

    def column_index(self, label):
        """Return the index of a column by looking up its label."""
        return self.labels.index(label)

    def apply(self, fn, *column_or_columns):
        """Apply ``fn`` to each element or elements of ``column_or_columns``.
        If no ``column_or_columns`` provided, `fn`` is applied to each row.

        Args:
            ``fn`` (function) -- The function to apply to each element
                of ``column_or_columns``.
            ``column_or_columns`` -- Columns containing the arguments to ``fn``
                as either column labels (``str``) or column indices (``int``).
                The number of columns must match the number of arguments
                that ``fn`` expects.

        Raises:
            ``ValueError`` -- if  ``column_label`` is not an existing
                column in the table.
            ``TypeError`` -- if insufficent number of ``column_label`` passed
                to ``fn``.

        Returns:
            An array consisting of results of applying ``fn`` to elements
            specified by ``column_label`` in each row.

        >>> t = Table().with_columns(
        ...     'letter', make_array('a', 'b', 'c', 'z'),
        ...     'count',  make_array(9, 3, 3, 1),
        ...     'points', make_array(1, 2, 2, 10))
        >>> t
        letter | count | points
        a      | 9     | 1
        b      | 3     | 2
        c      | 3     | 2
        z      | 1     | 10
        >>> t.apply(lambda x: x - 1, 'points')
        array([0, 1, 1, 9])
        >>> t.apply(lambda x, y: x * y, 'count', 'points')
        array([ 9,  6,  6, 10])
        >>> t.apply(lambda x: x - 1, 'count', 'points')
        Traceback (most recent call last):
            ...
        TypeError: <lambda>() takes 1 positional argument but 2 were given
        >>> t.apply(lambda x: x - 1, 'counts')
        Traceback (most recent call last):
            ...
        ValueError: The column "counts" is not in the table. The table contains these columns: letter, count, points

        Whole rows are passed to the function if no columns are specified.

        >>> t.apply(lambda row: row[1] * 2)
        array([18,  6,  6,  2])
        """
        if not column_or_columns:
            return np.array([fn(row) for row in self.rows])
        else:
            if len(column_or_columns) == 1 and \
                    _is_non_string_iterable(column_or_columns[0]):
                warnings.warn(
                   "column lists are deprecated; pass each as an argument", FutureWarning)
                column_or_columns = column_or_columns[0]
            rows = zip(*self.select(*column_or_columns).columns)
            return np.array([fn(*row) for row in rows])

    def first(self, label):
        """Return the zeroth item in a column."""
        return self.column(label)[0]

    def last(self, label):
        """Return the last item in a column."""
        return self.column(label)[-1]

    ############
    # Mutation #
    ############

    def set_format(self, column_or_columns, formatter):
        """Set the format of a column."""
        if inspect.isclass(formatter):
            formatter = formatter()
        if callable(formatter) and not hasattr(formatter, 'format_column'):
            formatter = _formats.FunctionFormatter(formatter)
        if not hasattr(formatter, 'format_column'):
            raise Exception('Expected Formatter or function: ' + str(formatter))
        for label in self._as_labels(column_or_columns):
            if formatter.converts_values:
                self[label] = formatter.convert_column(self[label])
            self._formats[label] = formatter
        return self

    def move_to_start(self, column_label):
        """Move a column to the first in order."""
        self._columns.move_to_end(self._as_label(column_label), last=False)
        return self

    def move_to_end(self, column_label):
        """Move a column to the last in order."""
        self._columns.move_to_end(self._as_label(column_label))
        return self

    def append(self, row_or_table):
        """Append a row or all rows of a table. An appended table must have all
        columns of self."""
        if isinstance(row_or_table, np.ndarray):
            row_or_table = row_or_table.tolist()
        elif not row_or_table:
            return
        if isinstance(row_or_table, Table):
            t = row_or_table
            columns = list(t.select(self.labels)._columns.values())
            n = t.num_rows
        else:
            if (len(list(row_or_table)) != self.num_columns):
                raise Exception('Row should have '+ str(self.num_columns) + " columns")
            columns, n = [[value] for value in row_or_table], 1
        for i, column in enumerate(self._columns):
            if self.num_rows:
                self._columns[column] = np.append(self[column], columns[i])
            else:
                self._columns[column] = np.array(columns[i])
        self._num_rows += n
        return self

    def append_column(self, label, values, formatter=None):
        """Appends a column to the table or replaces a column.

        ``__setitem__`` is aliased to this method:
            ``table.append_column('new_col', make_array(1, 2, 3))`` is equivalent to
            ``table['new_col'] = make_array(1, 2, 3)``.

        Args:
            ``label`` (str): The label of the new column.

            ``values`` (single value or list/array): If a single value, every
                value in the new column is ``values``.

                If a list or array, the new column contains the values in
                ``values``, which must be the same length as the table.
            ``formatter`` (single formatter): Adds a formatter to the column being 
                appended. No formatter added by default.

        Returns:
            Original table with new or replaced column

        Raises:
            ``ValueError``: If
                - ``label`` is not a string.
                - ``values`` is a list/array and does not have the same length
                  as the number of rows in the table.

        >>> table = Table().with_columns(
        ...     'letter', make_array('a', 'b', 'c', 'z'),
        ...     'count',  make_array(9, 3, 3, 1),
        ...     'points', make_array(1, 2, 2, 10))
        >>> table
        letter | count | points
        a      | 9     | 1
        b      | 3     | 2
        c      | 3     | 2
        z      | 1     | 10
        >>> table.append_column('new_col1', make_array(10, 20, 30, 40))
        letter | count | points | new_col1
        a      | 9     | 1      | 10
        b      | 3     | 2      | 20
        c      | 3     | 2      | 30
        z      | 1     | 10     | 40
        >>> table.append_column('new_col2', 'hello')
        letter | count | points | new_col1 | new_col2
        a      | 9     | 1      | 10       | hello
        b      | 3     | 2      | 20       | hello
        c      | 3     | 2      | 30       | hello
        z      | 1     | 10     | 40       | hello
        >>> table.append_column(123, make_array(1, 2, 3, 4))
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

        if (formatter != None):
            self.set_format(label, formatter)
        return self

    def relabel(self, column_label, new_label):
        """Changes the label(s) of column(s) specified by ``column_label`` to
        labels in ``new_label``.

        Args:
            ``column_label`` -- (single str or array of str) The label(s) of
                columns to be changed to ``new_label``.

            ``new_label`` -- (single str or array of str): The label name(s)
                of columns to replace ``column_label``.

        Raises:
            ``ValueError`` -- if ``column_label`` is not in table, or if
                ``column_label`` and ``new_label`` are not of equal length.
            ``TypeError`` -- if ``column_label`` and/or ``new_label`` is not
                ``str``.

        Returns:
            Original table with ``new_label`` in place of ``column_label``.

        >>> table = Table().with_columns(
        ...     'points', make_array(1, 2, 3),
        ...     'id',     make_array(12345, 123, 5123))
        >>> table.relabel('id', 'yolo')
        points | yolo
        1      | 12345
        2      | 123
        3      | 5123
        >>> table.relabel(make_array('points', 'yolo'),
        ...   make_array('red', 'blue'))
        red  | blue
        1    | 12345
        2    | 123
        3    | 5123
        >>> table.relabel(make_array('red', 'green', 'blue'),
        ...   make_array('cyan', 'magenta', 'yellow', 'key'))
        Traceback (most recent call last):
            ...
        ValueError: Invalid arguments. column_label and new_label must be of equal length.
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

    def remove(self, row_or_row_indices):
        """Removes a row or multiple rows of a table in place."""
        if not row_or_row_indices:
            return
        if isinstance(row_or_row_indices, int):
            rows_remove = [row_or_row_indices]
        else:
            rows_remove = row_or_row_indices
        for col in self._columns:
            self._columns[col] = [elem for i, elem in enumerate(self[col]) if i not in rows_remove]
        return self


    ##################
    # Transformation #
    ##################

    def copy(self, *, shallow=False):
        """Return a copy of a table."""
        table = type(self)()
        for label in self.labels:
            if shallow:
                column = self[label]
            else:
                column = np.copy(self[label])
            self._add_column_and_format(table, label, column)
        return table

    def select(self, *column_or_columns):
        """Return a table with only the columns in ``column_or_columns``.

        Args:
            ``column_or_columns``: Columns to select from the ``Table`` as
            either column labels (``str``) or column indices (``int``).

        Returns:
            A new instance of ``Table`` containing only selected columns.
            The columns of the new ``Table`` are in the order given in
            ``column_or_columns``.

        Raises:
            ``KeyError`` if any of ``column_or_columns`` are not in the table.

        >>> flowers = Table().with_columns(
        ...     'Number of petals', make_array(8, 34, 5),
        ...     'Name', make_array('lotus', 'sunflower', 'rose'),
        ...     'Weight', make_array(10, 5, 6)
        ... )

        >>> flowers
        Number of petals | Name      | Weight
        8                | lotus     | 10
        34               | sunflower | 5
        5                | rose      | 6

        >>> flowers.select('Number of petals', 'Weight')
        Number of petals | Weight
        8                | 10
        34               | 5
        5                | 6

        >>> flowers  # original table unchanged
        Number of petals | Name      | Weight
        8                | lotus     | 10
        34               | sunflower | 5
        5                | rose      | 6

        >>> flowers.select(0, 2)
        Number of petals | Weight
        8                | 10
        34               | 5
        5                | 6
        """
        labels = self._varargs_as_labels(column_or_columns)
        table = type(self)()
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

    def drop(self, *column_or_columns):
        """Return a Table with only columns other than selected label or
        labels.

        Args:
            ``column_or_columns`` (string or list of strings): The header
            names or indices of the columns to be dropped.

            ``column_or_columns`` must be an existing header name, or a
            valid column index.

        Returns:
            An instance of ``Table`` with given columns removed.

        >>> t = Table().with_columns(
        ...     'burgers',  make_array('cheeseburger', 'hamburger', 'veggie burger'),
        ...     'prices',   make_array(6, 5, 5),
        ...     'calories', make_array(743, 651, 582))
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
        >>> t.drop('burgers', 'calories')
        prices
        6
        5
        5
        >>> t.drop([0, 2])
        prices
        6
        5
        5
        >>> t.drop(0, 2)
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
        exclude = _varargs_labels_as_list(column_or_columns)
        return self.select([c for (i, c) in enumerate(self.labels)
                            if i not in exclude and c not in exclude])

    def where(self, column_or_label, value_or_predicate=None, other=None):
        """
        Return a new ``Table`` containing rows where ``value_or_predicate``
        returns True for values in ``column_or_label``.

        Args:
            ``column_or_label``: A column of the ``Table`` either as a label
            (``str``) or an index (``int``). Can also be an array of booleans;
            only the rows where the array value is ``True`` are kept.

            ``value_or_predicate``: If a function, it is applied to every value
            in ``column_or_label``. Only the rows where ``value_or_predicate``
            returns True are kept. If a single value, only the rows where the
            values in ``column_or_label`` are equal to ``value_or_predicate``
            are kept.

            ``other``: Optional additional column label for
            ``value_or_predicate`` to make pairwise comparisons. See the
            examples below for usage. When ``other`` is supplied,
            ``value_or_predicate`` must be a callable function.

        Returns:
            If ``value_or_predicate`` is a function, returns a new ``Table``
            containing only the rows where ``value_or_predicate(val)`` is True
            for the ``val``s in ``column_or_label``.

            If ``value_or_predicate`` is a value, returns a new ``Table``
            containing only the rows where the values in ``column_or_label``
            are equal to ``value_or_predicate``.

            If ``column_or_label`` is an array of booleans, returns a new
            ``Table`` containing only the rows where ``column_or_label`` is
            ``True``.

        >>> marbles = Table().with_columns(
        ...    "Color", make_array("Red", "Green", "Blue",
        ...                        "Red", "Green", "Green"),
        ...    "Shape", make_array("Round", "Rectangular", "Rectangular",
        ...                        "Round", "Rectangular", "Round"),
        ...    "Amount", make_array(4, 6, 12, 7, 9, 2),
        ...    "Price", make_array(1.30, 1.20, 2.00, 1.75, 0, 3.00))

        >>> marbles
        Color | Shape       | Amount | Price
        Red   | Round       | 4      | 1.3
        Green | Rectangular | 6      | 1.2
        Blue  | Rectangular | 12     | 2
        Red   | Round       | 7      | 1.75
        Green | Rectangular | 9      | 0
        Green | Round       | 2      | 3

        Use a value to select matching rows

        >>> marbles.where("Price", 1.3)
        Color | Shape | Amount | Price
        Red   | Round | 4      | 1.3

        In general, a higher order predicate function such as the functions in
        ``datascience.predicates.are`` can be used.

        >>> from datascience.predicates import are
        >>> # equivalent to previous example
        >>> marbles.where("Price", are.equal_to(1.3))
        Color | Shape | Amount | Price
        Red   | Round | 4      | 1.3

        >>> marbles.where("Price", are.above(1.5))
        Color | Shape       | Amount | Price
        Blue  | Rectangular | 12     | 2
        Red   | Round       | 7      | 1.75
        Green | Round       | 2      | 3

        Use the optional argument ``other`` to apply predicates to compare
        columns.

        >>> marbles.where("Price", are.above, "Amount")
        Color | Shape | Amount | Price
        Green | Round | 2      | 3

        >>> marbles.where("Price", are.equal_to, "Amount") # empty table
        Color | Shape | Amount | Price
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

            ``distinct``: if True, repeated values in ``column_or_label`` will
                be omitted.

        Returns:
            An instance of ``Table`` containing rows sorted based on the values
            in ``column_or_label``.

        >>> marbles = Table().with_columns(
        ...    "Color", make_array("Red", "Green", "Blue", "Red", "Green", "Green"),
        ...    "Shape", make_array("Round", "Rectangular", "Rectangular", "Round", "Rectangular", "Round"),
        ...    "Amount", make_array(4, 6, 12, 7, 9, 2),
        ...    "Price", make_array(1.30, 1.30, 2.00, 1.75, 1.40, 1.00))
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

        >>> marbles = Table().with_columns(
        ...    "Color", make_array("Red", "Green", "Blue", "Red", "Green", "Green"),
        ...    "Shape", make_array("Round", "Rectangular", "Rectangular", "Round", "Rectangular", "Round"),
        ...    "Amount", make_array(4, 6, 12, 7, 9, 2),
        ...    "Price", make_array(1.30, 1.30, 2.00, 1.75, 1.40, 1.00))
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
        # Assume that a call to group with a list of labels is a call to groups
        if _is_non_string_iterable(column_or_label) and \
                len(column_or_label) != self._num_rows:
            return self.groups(column_or_label, collect)

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

        >>> marbles = Table().with_columns(
        ...    "Color", make_array("Red", "Green", "Blue", "Red", "Green", "Green"),
        ...    "Shape", make_array("Round", "Rectangular", "Rectangular", "Round", "Rectangular", "Round"),
        ...    "Amount", make_array(4, 6, 12, 7, 9, 2),
        ...    "Price", make_array(1.30, 1.30, 2.00, 1.75, 1.40, 1.00))
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
        # Assume that a call to groups with one label is a call to group
        if not _is_non_string_iterable(labels):
            return self.group(labels, collect=collect)

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
        """Generate a table with a column for each unique value in ``columns``,
        with rows for each unique value in ``rows``. Each row counts/aggregates
        the values that match both row and column based on ``collect``.

        Args:
            ``columns`` -- a single column label or index, (``str`` or ``int``),
                used to create new columns, based on its unique values.
            ``rows`` -- row labels or indices, (``str`` or ``int`` or list),
                used to create new rows based on it's unique values.
            ``values`` -- column label in table for use in aggregation.
                Default None.
            ``collect`` -- aggregation function, used to group ``values``
                over row-column combinations. Default None.
            ``zero`` -- zero value to use for non-existent row-column
                combinations.

        Raises:
            TypeError -- if ``collect`` is passed in and ``values`` is not,
                vice versa.

        Returns:
            New pivot table, with row-column combinations, as specified, with
            aggregated ``values`` by ``collect`` across the intersection of
            ``columns`` and ``rows``. Simple counts provided if values and
            collect are None, as default.

        >>> titanic = Table().with_columns('age', make_array(21, 44, 56, 89, 95
        ...    , 40, 80, 45), 'survival', make_array(0,0,0,1, 1, 1, 0, 1),
        ...    'gender',  make_array('M', 'M', 'M', 'M', 'F', 'F', 'F', 'F'),
        ...    'prediction', make_array(0, 0, 1, 1, 0, 1, 0, 1))
        >>> titanic
        age  | survival | gender | prediction
        21   | 0        | M      | 0
        44   | 0        | M      | 0
        56   | 0        | M      | 1
        89   | 1        | M      | 1
        95   | 1        | F      | 0
        40   | 1        | F      | 1
        80   | 0        | F      | 0
        45   | 1        | F      | 1
        >>> titanic.pivot('survival', 'gender')
        gender | 0    | 1
        F      | 1    | 3
        M      | 3    | 1
        >>> titanic.pivot('prediction', 'gender')
        gender | 0    | 1
        F      | 2    | 2
        M      | 2    | 2
        >>> titanic.pivot('survival', 'gender', values='age', collect = np.mean)
        gender | 0       | 1
        F      | 80      | 60
        M      | 40.3333 | 89
        >>> titanic.pivot('survival', make_array('prediction', 'gender'))
        prediction | gender | 0    | 1
        0          | F      | 1    | 1
        0          | M      | 2    | 0
        1          | F      | 0    | 2
        1          | M      | 1    | 1
        >>> titanic.pivot('survival', 'gender', values = 'age')
        Traceback (most recent call last):
           ...
        TypeError: values requires collect to be specified
        >>> titanic.pivot('survival', 'gender', collect = np.mean)
        Traceback (most recent call last):
           ...
        TypeError: collect requires values to be specified
        """
        if collect is not None and values is None:
            raise TypeError('collect requires values to be specified')
        if values is not None and collect is None:
            raise TypeError('values requires collect to be specified')
        columns = self._as_label(columns)
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
        binned = type(self)().with_column('bin',rbins)
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
        """Creates a new table with the columns of self and other, containing
        rows for all values of a column that appear in both tables.

        Args:
            ``column_label``:  label of column or array of labels in self that is used to
                join  rows of ``other``.
            ``other``: Table object to join with self on matching values of
                ``column_label``.

        Kwargs:
            ``other_label``: default None, assumes ``column_label``.
                Otherwise in ``other`` used to join rows.

        Returns:
            New table self joined with ``other`` by matching values in
            ``column_label`` and ``other_label``. If the resulting join is
            empty, returns None.

        >>> table = Table().with_columns('a', make_array(9, 3, 3, 1),
        ...     'b', make_array(1, 2, 2, 10),
        ...     'c', make_array(3, 4, 5, 6))
        >>> table
        a    | b    | c
        9    | 1    | 3
        3    | 2    | 4
        3    | 2    | 5
        1    | 10   | 6
        >>> table2 = Table().with_columns( 'a', make_array(9, 1, 1, 1),
        ... 'd', make_array(1, 2, 2, 10),
        ... 'e', make_array(3, 4, 5, 6))
        >>> table2
        a    | d    | e
        9    | 1    | 3
        1    | 2    | 4
        1    | 2    | 5
        1    | 10   | 6
        >>> table.join('a', table2)
        a    | b    | c    | d    | e
        1    | 10   | 6    | 2    | 4
        1    | 10   | 6    | 2    | 5
        1    | 10   | 6    | 10   | 6
        9    | 1    | 3    | 1    | 3
        >>> table.join('a', table2, 'a') # Equivalent to previous join
        a    | b    | c    | d    | e
        1    | 10   | 6    | 2    | 4
        1    | 10   | 6    | 2    | 5
        1    | 10   | 6    | 10   | 6
        9    | 1    | 3    | 1    | 3
        >>> table.join('a', table2, 'd') # Repeat column labels relabeled
        a    | b    | c    | a_2  | e
        1    | 10   | 6    | 9    | 3
        >>> table2 #table2 has three rows with a = 1
        a    | d    | e
        9    | 1    | 3
        1    | 2    | 4
        1    | 2    | 5
        1    | 10   | 6
        >>> table #table has only one row with a = 1
        a    | b    | c
        9    | 1    | 3
        3    | 2    | 4
        3    | 2    | 5
        1    | 10   | 6
        >>> table.join(['a', 'b'], table2, ['a', 'd']) # joining on multiple columns
        a    | b    | c    | e
        1    | 10   | 6    | 6
        9    | 1    | 3    | 3
        """
        if self.num_rows == 0 or other.num_rows == 0:
            return None
        if not other_label:
            other_label = column_label

        # checking to see if joining on multiple columns
        if _is_non_string_iterable(column_label):
            # then we are going to be joining multiple labels
            return self._multiple_join(column_label, other, other_label)

        # original single column join
        return self._join(column_label, other, other_label)

    def _join(self, column_label, other, other_label=[]):
        """joins when COLUMN_LABEL is a string"""
        if self.num_rows == 0 or other.num_rows == 0:
            return None
        if not other_label:
            other_label = column_label

        self_rows = self.index_by(column_label)
        other_rows = other.index_by(other_label)
        return self._join_helper([column_label], self_rows, other, [other_label], other_rows)

    def _multiple_join(self, column_label, other, other_label=[]):
        """joins when column_label is a non-string iterable"""
        assert len(column_label) == len(other_label), 'unequal number of columns'

        self_rows = self._multi_index(column_label)
        other_rows = other._multi_index(other_label)
        return self._join_helper(column_label, self_rows, other, other_label, other_rows)
        

    def _join_helper(self, column_label, self_rows, other, other_label, other_rows):
        # Gather joined rows from self_rows that have join values in other_rows
        joined_rows = []
        for v, rows in self_rows.items():
            if v in other_rows:
                joined_rows += [row + o for row in rows for o in other_rows[v]]
        if not joined_rows:
            return None

        # Build joined table
        self_labels = list(self.labels)
        other_labels = [self._unused_label(s) for s in other.labels]
        if (len(set(self_labels + other_labels)) != len(list(self_labels + other_labels))):
            other_labels = [self._unused_label_in_either_table(s, other) for s in other.labels]
        other_labels_map = dict(zip(other.labels, other_labels))
        joined = type(self)(self_labels + other_labels).with_rows(joined_rows)

        # Copy formats from both tables
        joined._formats.update(self._formats)
        for label in other._formats:
            joined._formats[other_labels_map[label]] = other._formats[label]

        # Remove redundant column, but perhaps save its formatting
        for duplicate in other_label:
            del joined[other_labels_map[duplicate]]
        for duplicate in other_label:
            if duplicate not in self._formats and duplicate in other._formats:
                joined._formats[duplicate] = other._formats[duplicate]

        for col in column_label[::-1]:
            joined = joined.move_to_start(col).sort(col)

        return joined

    def stats(self, ops=(min, max, np.median, sum)):
        """Compute statistics for each column and place them in a table."""
        names = [op.__name__ for op in ops]
        ops = [_zero_on_type_error(op) for op in ops]
        columns = [[op(column) for op in ops] for column in self.columns]
        table = type(self)().with_columns(zip(self.labels, columns))
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

    def _varargs_as_labels(self, label_list):
        """Converts a list of labels or singleton list of list of labels into
        a list of labels.  Useful when labels are passed as varargs."""
        return self._as_labels(_varargs_labels_as_list(label_list))

    def _unused_label(self, label):
        """Generate an unused label."""
        original = label
        existing = self.labels
        i = 2
        while label in existing:
            label = '{}_{}'.format(original, i)
            i += 1
        return label

    def _unused_label_in_either_table(self, label, other):
        original = label
        existing_self = self.labels
        existing_other = other.labels
        i = 2
        while label in existing_self:
            label = '{}_{}'.format(original, i)
            i += 1
            while label in existing_other:
                label = '{}_{}'.format(original, i)
                i += 1
        return label

    def _get_column(self, column_or_label):
        """Convert label to column and check column length."""
        c = column_or_label
        if isinstance(c, collections.abc.Hashable) and c in self.labels:
            return self[c]
        elif isinstance(c, numbers.Integral):
            return self[c]
        elif isinstance(c, str):
            raise ValueError('label "{}" not in labels {}'.format(c, self.labels))
        else:
            assert len(c) == self.num_rows, 'column length mismatch'
            return c

    def percentile(self, p):
        """Return a new table with one row containing the pth percentile for
        each column.

        Assumes that each column only contains one type of value.

        Returns a new table with one row and the same column labels.
        The row contains the pth percentile of the original column, where the
        pth percentile of a column is the smallest value that at at least as
        large as the p% of numbers in the column.

        >>> table = Table().with_columns(
        ...     'count',  make_array(9, 3, 3, 1),
        ...     'points', make_array(1, 2, 2, 10))
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

    def sample(self, k=None, with_replacement=True, weights=None):
        """Return a new table where k rows are randomly sampled from the
        original table.

        Args:
            ``k`` -- specifies the number of rows (``int``) to be sampled from
               the table. Default is k equal to number of rows in the table.

            ``with_replacement`` -- (``bool``) By default True;
                Samples ``k`` rows with replacement from table, else samples
                ``k`` rows without replacement.

            ``weights`` -- Array specifying probability the ith row of the
                table is sampled. Defaults to None, which samples each row
                with equal probability. ``weights`` must be a valid probability
                distribution -- i.e. an array the length of the number of rows,
                summing to 1.

        Raises:
            ValueError -- if ``weights`` is not length equal to number of rows
                in the table; or, if ``weights`` does not sum to 1.

        Returns:
            A new instance of ``Table`` with ``k`` rows resampled.

        >>> jobs = Table().with_columns(
        ...     'job',  make_array('a', 'b', 'c', 'd'),
        ...     'wage', make_array(10, 20, 15, 8))
        >>> jobs
        job  | wage
        a    | 10
        b    | 20
        c    | 15
        d    | 8
        >>> jobs.sample() # doctest: +SKIP
        job  | wage
        b    | 20
        b    | 20
        a    | 10
        d    | 8
        >>> jobs.sample(with_replacement=True) # doctest: +SKIP
        job  | wage
        d    | 8
        b    | 20
        c    | 15
        a    | 10
        >>> jobs.sample(k = 2) # doctest: +SKIP
        job  | wage
        b    | 20
        c    | 15
        >>> ws =  make_array(0.5, 0.5, 0, 0)
        >>> jobs.sample(k=2, with_replacement=True, weights=ws) # doctest: +SKIP
        job  | wage
        a    | 10
        a    | 10
        >>> jobs.sample(k=2, weights=make_array(1, 0, 1, 0))
        Traceback (most recent call last):
            ...
        ValueError: probabilities do not sum to 1
        >>> jobs.sample(k=2, weights=make_array(1, 0, 0)) # Weights must be length of table.
        Traceback (most recent call last):
            ...
        ValueError: 'a' and 'p' must have same size
        """
        n = self.num_rows
        if k is None:
            k = n
        index = np.random.choice(n, k, replace=with_replacement, p=weights)
        columns = [[c[i] for i in index] for c in self.columns]
        sample = self._with_columns(columns)
        return sample

    def shuffle(self):
        """Return a new table where all the rows are randomly shuffled from the
        original table..

        Returns:
            A new instance of ``Table`` with all ``k`` rows shuffled.
        """
        return self.sample(with_replacement=False)

    def sample_from_distribution(self, distribution, k, proportions=False):
        """Return a new table with the same number of rows and a new column.
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
        """Return a tuple of two tables where the first table contains
        ``k`` rows randomly sampled and the second contains the remaining rows.

        Args:
            ``k`` (int): The number of rows randomly sampled into the first
                table. ``k`` must be between 1 and ``num_rows - 1``.

        Raises:
            ``ValueError``: ``k`` is not between 1 and ``num_rows - 1``.

        Returns:
            A tuple containing two instances of ``Table``.

        >>> jobs = Table().with_columns(
        ...     'job',  make_array('a', 'b', 'c', 'd'),
        ...     'wage', make_array(10, 20, 15, 8))
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

        rows = np.random.permutation(self.num_rows)

        first = self.take(rows[:k])
        rest = self.take(rows[k:])
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

        >>> tiles = Table(make_array('letter', 'count', 'points'))
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

        >>> tiles = Table(make_array('letter', 'count', 'points'))
        >>> tiles.with_rows(make_array(make_array('c', 2, 3),
        ...     make_array('d', 4, 2)))
        letter | count | points
        c      | 2     | 3
        d      | 4     | 2
        """
        self = self.copy()
        self.append(self._with_columns(zip(*rows)))
        return self

    def with_column(self, label, values, formatter=None):
        """Return a new table with an additional or replaced column.

        Args:
            ``label`` (str): The column label. If an existing label is used,
                the existing column will be replaced in the new table.

            ``values`` (single value or sequence): If a single value, every
                value in the new column is ``values``. If sequence of values,
                new column takes on values in ``values``.

            ``formatter`` (single value): Specifies formatter for the new column. Defaults to no formatter.

        Raises:
            ``ValueError``: If
                - ``label`` is not a valid column name
                - if ``label`` is not of type (str)
                - ``values`` is a list/array that does not have the same
                    length as the number of rows in the table.

        Returns:
            copy of original table with new or replaced column

        >>> alphabet = Table().with_column('letter', make_array('c','d'))
        >>> alphabet = alphabet.with_column('count', make_array(2, 4))
        >>> alphabet
        letter | count
        c      | 2
        d      | 4
        >>> alphabet.with_column('permutes', make_array('a', 'g'))
        letter | count | permutes
        c      | 2     | a
        d      | 4     | g
        >>> alphabet
        letter | count
        c      | 2
        d      | 4
        >>> alphabet.with_column('count', 1)
        letter | count
        c      | 1
        d      | 1
        >>> alphabet.with_column(1, make_array(1, 2))
        Traceback (most recent call last):
            ...
        ValueError: The column label must be a string, but a int was given
        >>> alphabet.with_column('bad_col', make_array(1))
        Traceback (most recent call last):
            ...
        ValueError: Column length mismatch. New column does not have the same number of rows as table.
        """
        # Ensure that if with_column is called instead of with_columns;
        # no error is raised.

        new_table = self.copy()
        if formatter == {}:
            formatter = None
        elif isinstance(formatter, dict):
            formatter = formatter["formatter"]
        new_table.append_column(label, values, formatter)
        return new_table

    def with_columns(self, *labels_and_values, **formatter):
        """Return a table with additional or replaced columns.


        Args:
            ``labels_and_values``: An alternating list of labels and values 
                or a list of label-value pairs. If one of the labels is in 
                existing table, then every value in the corresponding column is
                set to that value. If label has only a single value (``int``),
                every row of corresponding column takes on that value.
            ''formatter'' (single Formatter value): A single formatter value 
                that will be applied to all columns being added using this 
                function call.

        Raises:
            ``ValueError``: If
                - any label in ``labels_and_values`` is not a valid column
                    name, i.e if label is not of type (str).
                - if any value in ``labels_and_values`` is a list/array and
                    does not have the same length as the number of rows in the
                    table.
            ``AssertionError``:
                - 'incorrect columns format', if passed more than one sequence
                    (iterables) for  ``labels_and_values``.
                - 'even length sequence required' if missing a pair in
                    label-value pairs.


        Returns:
            Copy of original table with new or replaced columns. Columns added
            in order of labels. Equivalent to ``with_column(label, value)``
            when passed only one label-value pair.


        >>> players = Table().with_columns('player_id',
        ...     make_array(110234, 110235), 'wOBA', make_array(.354, .236))
        >>> players
        player_id | wOBA
        110234    | 0.354
        110235    | 0.236
        >>> players = players.with_columns('salaries', 'N/A', 'season', 2016)
        >>> players
        player_id | wOBA  | salaries | season
        110234    | 0.354 | N/A      | 2016
        110235    | 0.236 | N/A      | 2016
        >>> salaries = Table().with_column('salary',
        ...     make_array(500000, 15500000))
        >>> players.with_columns('salaries', salaries.column('salary'),
        ...     'bonus', make_array(6, 1), formatter=_formats.CurrencyFormatter)
        player_id | wOBA  | salaries    | season | bonus
        110234    | 0.354 | $500,000    | 2016   | $6
        110235    | 0.236 | $15,500,000 | 2016   | $1
        >>> players.with_columns(2, make_array('$600,000', '$20,000,000'))
        Traceback (most recent call last):
            ...
        ValueError: The column label must be a string, but a int was given
        >>> players.with_columns('salaries', make_array('$600,000'))
        Traceback (most recent call last):
            ...
        ValueError: Column length mismatch. New column does not have the same number of rows as table.
        """
        if len(labels_and_values) == 1:
            labels_and_values = labels_and_values[0]
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
            self = self.with_column(label, values, formatter)
        return self

        

    def relabeled(self, label, new_label):
        """Return a new table with ``label`` specifying column label(s)
        replaced by corresponding ``new_label``.

        Args:
            ``label`` -- (str or array of str) The label(s) of
                columns to be changed.

            ``new_label`` -- (str or array of str): The new label(s) of
                columns to be changed. Same number of elements as label.

        Raises:
            ``ValueError`` -- if ``label`` does not exist in
                table, or if the ``label`` and ``new_label`` are not not of
                equal length. Also, raised if ``label`` and/or ``new_label``
                are not ``str``.

        Returns:
            New table with ``new_label`` in place of ``label``.

        >>> tiles = Table().with_columns('letter', make_array('c', 'd'),
        ...    'count', make_array(2, 4))
        >>> tiles
        letter | count
        c      | 2
        d      | 4
        >>> tiles.relabeled('count', 'number')
        letter | number
        c      | 2
        d      | 4
        >>> tiles  # original table unmodified
        letter | count
        c      | 2
        d      | 4
        >>> tiles.relabeled(make_array('letter', 'count'),
        ...   make_array('column1', 'column2'))
        column1 | column2
        c       | 2
        d       | 4
        >>> tiles.relabeled(make_array('letter', 'number'),
        ...  make_array('column1', 'column2'))
        Traceback (most recent call last):
            ...
        ValueError: Invalid labels. Column labels must already exist in table in order to be replaced.
        """
        copy = self.copy()
        copy.relabel(label, new_label)
        return copy

    # Deprecated
    def with_relabeling(self, *args):
        warnings.warn("with_relabeling is deprecated; use relabeled", FutureWarning)
        return self.relabeled(*args)

    def bin(self, *columns, **vargs):
        """Group values by bin and compute counts per bin by column.

        By default, bins are chosen to contain all values in all columns. The
        following named arguments from numpy.histogram can be applied to
        specialize bin widths:

        If the original table has n columns, the resulting binned table has
        n+1 columns, where column 0 contains the lower bound of each bin.

        Args:
            ``columns`` (str or int): Labels or indices of columns to be
                binned. If empty, all columns are binned.

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
        if columns:
            self = self.select(*columns)
        if 'normed' in vargs:
            vargs.setdefault('density', vargs.pop('normed'))
        density = vargs.get('density', False)
        tag = 'density' if density else 'count'

        cols = list(self._columns.values())
        _, bins = np.histogram(cols, **vargs)

        binned = type(self)().with_column('bin', bins)
        for label in self.labels:
            counts, _ = np.histogram(self[label], bins=bins, density=density)
            binned[label + ' ' + tag] = np.append(counts, 0)
        return binned

    def move_column(self, label, index):
        """Returns a new table with specified column moved to the specified column index."""
        table = type(self)()
        col_order = list(self._columns)
        label_idx = col_order.index(self._as_label(label))
        col_to_move = col_order.pop(label_idx)
        col_order.insert(index, col_to_move)
        for col in col_order:
            table[col] = self[col]
        return table

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

    @staticmethod
    def _use_html_if_available(format_fn):
        """Use the value's HTML rendering if available, overriding format_fn."""
        def format_using_as_html(v, label=False):
            if not label and hasattr(v, 'as_html'):
                return v.as_html()
            else:
                return format_fn(v, label)
        return format_using_as_html

    def _get_column_formatters(self, max_rows, as_html):
        """Return one value formatting function per column.

        Each function has the signature f(value, label=False) -> str
        """
        formats = {s: self._formats.get(s, self.formatter) for s in self.labels}
        cols = self._columns.items()
        fmts = [formats[k].format_column(k, v[:max_rows]) for k, v in cols]
        if as_html:
            fmts = list(map(type(self)._use_html_if_available, fmts))
        return fmts

    def as_text(self, max_rows=0, sep=" | "):
        """Format table as text."""
        if not max_rows or max_rows > self.num_rows:
            max_rows = self.num_rows
        omitted = max(0, self.num_rows - max_rows)
        labels = self._columns.keys()
        fmts = self._get_column_formatters(max_rows, False)
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
        fmts = self._get_column_formatters(max_rows, True)
        for row in itertools.islice(self.rows, max_rows):
            lines += [
                (2, '<tr>'),
                (3, ' '.join('<td>' + fmt(v, label=False) + '</td>' for
                    v, fmt in zip(row, fmts))),
                (2, '</tr>'),
            ]
        lines.append((1, '</tbody>'))
        lines.append((0, '</table>'))
        if omitted:
            lines.append((0, '<p>... ({} rows omitted)</p>'.format(omitted)))
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

    def _multi_index(self, columns_or_labels):
        """Returns a dict keyed by a tuple of the values that correspond to
        the selected COLUMNS_OR_LABELS, with values corresponding to """
        columns = [self._get_column(col) for col in columns_or_labels]
        index = {}
        for key, row in zip(zip(*columns), self.rows):
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

        >>> jobs = Table().with_columns(
        ...     'job',  make_array('a', 'b', 'c', 'd'),
        ...     'wage', make_array(10, 20, 15, 8))
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
        """Convert the table to a structured NumPy array."""
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
        (0.0, 30/256, 66/256),
        (1.0, 200/256, 44/256),
        (0.0, 150/256, 207/256),
        (30/256, 100/256, 0.0),
        (172/256, 60/256, 72/256),
    )
    chart_colors += tuple(tuple((x+0.7)/2 for x in c) for c in chart_colors)

    default_alpha = 0.7

    default_options = {
        'alpha': default_alpha,
    }

    def plot(self, column_for_xticks=None, select=None, overlay=True, width=6, height=4, **vargs):
        """Plot line charts for the table.

        Args:
            column_for_xticks (``str/array``): A column containing x-axis labels

        Kwargs:
            overlay (bool): create a chart with one color per data column;
                if False, each plot will be displayed separately.

            vargs: Additional arguments that get passed into `plt.plot`.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
                for additional arguments that can be passed into vargs.
        Raises:
            ValueError -- Every selected column must be numerical.

        Returns:
            Returns a line plot (connected scatter). Each plot is labeled using
            the values in `column_for_xticks` and one plot is produced for all
            other columns in self (or for the columns designated by `select`).

        >>> table = Table().with_columns(
        ...     'days',  make_array(0, 1, 2, 3, 4, 5),
        ...     'price', make_array(90.5, 90.00, 83.00, 95.50, 82.00, 82.00),
        ...     'projection', make_array(90.75, 82.00, 82.50, 82.50, 83.00, 82.50))
        >>> table
        days | price | projection
        0    | 90.5  | 90.75
        1    | 90    | 82
        2    | 83    | 82.5
        3    | 95.5  | 82.5
        4    | 82    | 83
        5    | 82    | 82.5
        >>> table.plot('days') # doctest: +SKIP
        <line graph with days as x-axis and lines for price and projection>
        >>> table.plot('days', overlay=False) # doctest: +SKIP
        <line graph with days as x-axis and line for price>
        <line graph with days as x-axis and line for projection>
        >>> table.plot('days', 'price') # doctest: +SKIP
        <line graph with days as x-axis and line for price>
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

        self._visualize(x_label, y_labels, None, overlay, draw, _vertical_x, width=width, height=height)

    def bar(self, column_for_categories=None, select=None, overlay=True, width=6, height=4, **vargs):
        """Plot bar charts for the table.

        Each plot is labeled using the values in `column_for_categories` and
        one plot is produced for every other column (or for the columns
        designated by `select`).

        Every selected column except `column_for_categories` must be numerical.

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

        # Matplotlib tries to center the labels, but we already handle that
        # TODO consider changing the custom centering code and using matplotlib's default
        vargs['align'] = 'edge'
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

        self._visualize(column_for_categories, labels, xticks, overlay, draw, annotate, width=width, height=height)


    def group_bar(self, column_label, **vargs):
        """Plot a bar chart for the table.

        The values of the specified column are grouped and counted, and one
        bar is produced for each group.

        Note: This differs from ``bar`` in that there is no need to specify
        bar heights; the height of a category's bar is the number of copies
        of that category in the given column.  This method behaves more like
        ``hist`` in that regard, while ``bar`` behaves more like ``plot`` or
        ``scatter`` (which require the height of each point to be specified).

        Args:
            ``column_label`` (str or int): The name or index of a column

        Kwargs:
            overlay (bool): create a chart with one color per data column;
                if False, each will be displayed separately.

            width (float): The width of the plot, in inches
            height (float): The height of the plot, in inches

            vargs: Additional arguments that get passed into `plt.bar`.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.bar
                for additional arguments that can be passed into vargs.
        """
        self.group(column_label).bar(column_label, **vargs)

    def barh(self, column_for_categories=None, select=None, overlay=True, width=6, **vargs):
        """Plot horizontal bar charts for the table.

        Args:
            ``column_for_categories`` (``str``): A column containing y-axis categories
                used to create buckets for bar chart.

        Kwargs:
            overlay (bool): create a chart with one color per data column;
                if False, each will be displayed separately.

            vargs: Additional arguments that get passed into `plt.barh`.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.barh
                for additional arguments that can be passed into vargs.

        Raises:
            ValueError -- Every selected except column for ``column_for_categories``
                must be numerical.

        Returns:
            Horizontal bar graph with buckets specified by ``column_for_categories``.
            Each plot is labeled using the values in ``column_for_categories``
            and one plot is produced for every other column (or for the columns
            designated by ``select``).

        >>> t = Table().with_columns(
        ...     'Furniture', make_array('chairs', 'tables', 'desks'),
        ...     'Count', make_array(6, 1, 2),
        ...     'Price', make_array(10, 20, 30)
        ...     )
        >>> t
        Furniture | Count | Price
        chairs    | 6     | 10
        tables    | 1     | 20
        desks     | 2     | 30
        >>> furniture_table.barh('Furniture') # doctest: +SKIP
        <bar graph with furniture as categories and bars for count and price>
        >>> furniture_table.barh('Furniture', 'Price') # doctest: +SKIP
        <bar graph with furniture as categories and bars for price>
        >>> furniture_table.barh('Furniture', make_array(1, 2)) # doctest: +SKIP
        <bar graph with furniture as categories and bars for count and price>
        """
        options = self.default_options.copy()
        # Matplotlib tries to center the labels, but we already handle that
        # TODO consider changing the custom centering code and using matplotlib's default
        vargs['align'] = 'edge'
        options.update(vargs)

        yticks, labels = self._split_column_and_labels(column_for_categories)
        if select is not None:
            labels = self._as_labels(select)
        n = len(labels)

        index = np.arange(self.num_rows)
        margin = 0.1
        bwidth = 1 - 2 * margin
        if overlay:
            bwidth /= len(labels)

        if 'height' in options:
            height = options.pop('height')
        else:
            height = max(4, len(index)/2)

        def draw(axis, label, color):
            if overlay:
                ypos = index + margin + (1-2*margin)*(n - 1 - labels.index(label))/n
            else:
                ypos = index
            # barh plots entries in reverse order from bottom to top
            axis.barh(ypos, self[label][::-1], bwidth,  color=color, **options)

        ylabel = self._as_label(column_for_categories)

        def annotate(axis, ticks):
            axis.set_yticks(index+0.5) # Center labels on bars
            # barh plots entries in reverse order from bottom to top
            axis.set_yticklabels(ticks[::-1], stretch='ultra-condensed')
            axis.set_xlabel(axis.get_ylabel())
            axis.set_ylabel(ylabel)

        self._visualize('', labels, yticks, overlay, draw, annotate, width=width, height=height)


    def group_barh(self, column_label, **vargs):
        """Plot a horizontal bar chart for the table.

        The values of the specified column are grouped and counted, and one
        bar is produced for each group.

        Note: This differs from ``barh`` in that there is no need to specify
        bar heights; the size of a category's bar is the number of copies
        of that category in the given column.  This method behaves more like
        ``hist`` in that regard, while ``barh`` behaves more like ``plot`` or
        ``scatter`` (which require the second coordinate of each point to be
        specified in another column).

        Args:
            ``column_label`` (str or int): The name or index of a column

        Kwargs:
            overlay (bool): create a chart with one color per data column;
                if False, each will be displayed separately.

            width (float): The width of the plot, in inches
            height (float): The height of the plot, in inches

            vargs: Additional arguments that get passed into `plt.bar`.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.bar
                for additional arguments that can be passed into vargs.
        """
        self.group(column_label).barh(column_label, **vargs)

    def scatter(self, column_for_x, select=None, overlay=True, fit_line=False,
        group=None, labels=None, sizes=None, width=5, height=5, s=20,
        colors=None, **vargs):
        """Creates scatterplots, optionally adding a line of best fit.

        Args:
            ``column_for_x`` (``str``): The column to use for the x-axis values
                and label of the scatter plots.

        Kwargs:
            ``overlay`` (``bool``): If true, creates a chart with one color
                per data column; if False, each plot will be displayed separately.

            ``fit_line`` (``bool``): draw a line of best fit for each set of points.

            ``vargs``: Additional arguments that get passed into `plt.scatter`.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter
                for additional arguments that can be passed into vargs. These
                include: `marker` and `norm`, to name a couple.

            ``group``: A column of categories to be used for coloring dots per 
                each category grouping.

            ``labels``: A column of text labels to annotate dots.

            ``sizes``:  A column of values to set the relative areas of dots.

            ``s``: Size of dots. If sizes is also provided, then dots will be
                in the range 0 to 2 * s.

            ``colors``: (deprecated) A synonym for ``group``. Retained
                temporarily for backwards compatibility. This argument
                will be removed in future releases.

        Raises:
            ValueError -- Every column, ``column_for_x`` or ``select``, must be numerical

        Returns:
            Scatter plot of values of ``column_for_x`` plotted against
            values for all other columns in self. Each plot uses the values in
            `column_for_x` for horizontal positions. One plot is produced for
            all other columns in self as y (or for the columns designated by
            `select`).


        >>> table = Table().with_columns(
        ...     'x', make_array(9, 3, 3, 1),
        ...     'y', make_array(1, 2, 2, 10),
        ...     'z', make_array(3, 4, 5, 6))
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
        if group is not None and colors is not None and group != colors:
            warnings.warn("Do not pass both colors and group to scatter().")
        if group is None and colors is not None:
            # Backward compatibility
            group = colors
            # TODO: In a future release, warn that this is deprecated.
            # Deprecated
            # warnings.warn("scatter(colors=x) is deprecated. Use scatter(group=x)", FutureWarning)
        if group is not None:
            y_labels.remove(self._as_label(group))
        if sizes is not None:
            y_labels.remove(self._as_label(sizes))
        if select is not None:
            y_labels = self._as_labels(select)
        if len(y_labels) > 1 and group is not None and overlay:
            warnings.warn("Group and overlay are incompatible in a scatter")
            overlay = False

        def draw(axis, label, color):
            if group is not None:
                colored = sorted(np.unique(self.column(group)))
                color_list = list(itertools.islice(itertools.cycle(self.chart_colors), len(colored)))
                color_map = collections.OrderedDict(zip(colored, color_list))
                color = [color_map[x] for x in self.column(group)]
            elif 'color' in options:
                color = options.pop('color')
            y_data = self[label]
            if sizes is not None:
                max_size = max(self[sizes]) ** 0.5
                size = 2 * s * self[sizes] ** 0.5 / max_size
            else:
                size = s
            axis.scatter(x_data, y_data, color=color, s=size, **options)
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
            if group is not None:
                import matplotlib.patches as mpatches
                group_col_name = self._as_label(group)
                patches = [mpatches.Patch(color=c, label="{0}={1}".format(group_col_name, v)) \
                    for (v, c) in color_map.items()]
                axis.legend(loc=2, bbox_to_anchor=(1.05, 1), handles=patches)

        x_label = self._as_label(column_for_x)
        self._visualize(x_label, y_labels, None, overlay, draw, _vertical_x, width=width, height=height)

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
            type(self).plots.append(axis)
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
                type(self).plots.append(axis)

    def _split_column_and_labels(self, column_or_label):
        """Return the specified column and labels of other columns."""
        column = None if column_or_label is None else self._get_column(column_or_label)
        labels = [label for i, label in enumerate(self.labels) if column_or_label not in (i, label)]
        return column, labels

    # Deprecated
    def pivot_hist(self, pivot_column_label, value_column_label, overlay=True, width=6, height=4, **vargs):
        """Draw histograms of each category in a column. (Deprecated)"""
        warnings.warn("pivot_hist is deprecated; use "
                      "hist(value_column_label, group=pivot_column_label), or "
                      "with side_by_side=True if you really want side-by-side "
                      "bars.")
        pvt_labels = np.unique(self[pivot_column_label])
        pvt_columns = [self[value_column_label][np.where(self[pivot_column_label] == pivot)] for pivot in pvt_labels]
        n = len(pvt_labels)
        colors = list(itertools.islice(itertools.cycle(self.chart_colors), n))
        if overlay:
            plt.figure(figsize=(width, height))
            vals, bins, patches = plt.hist(pvt_columns, color=colors, **vargs)
            plt.legend(pvt_labels)
        else:
            _, axes = plt.subplots(n, 1, figsize=(width, height * n))
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
        t = type(self)()
        t['start'] = bins[0:-1]
        t['end'] = bins[1:]
        for label, column in zip(pvt_labels,vals):
            t[label] = column

    def hist(self, *columns, overlay=True, bins=None, bin_column=None, unit=None, counts=None, group=None, side_by_side=False, width=6, height=4, **vargs):
        """Plots one histogram for each column in columns. If no column is
        specified, plot all columns.

        Kwargs:
            overlay (bool): If True, plots 1 chart with all the histograms
                overlaid on top of each other (instead of the default behavior
                of one histogram for each column in the table). Also adds a
                legend that matches each bar color to its column.  Note that
                if the histograms are not overlaid, they are not forced to the
                same scale.

            bins (list or int): Lower bound for each bin in the
                histogram or number of bins. If None, bins will
                be chosen automatically.

            bin_column (column name or index): A column of bin lower bounds.
                All other columns are treated as counts of these bins.
                If None, each value in each row is assigned a count of 1.

            counts (column name or index): Deprecated name for bin_column.

            unit (string): A name for the units of the plotted column (e.g.
                'kg'), to be used in the plot.

            group (column name or index): A column of categories.  The rows are
                grouped by the values in this column, and a separate histogram is
                generated for each group.  The histograms are overlaid or plotted
                separately depending on the overlay argument.  If None, no such
                grouping is done.

            side_by_side (bool): Whether histogram bins should be plotted side by
                side (instead of directly overlaid).  Makes sense only when
                plotting multiple histograms, either by passing several columns
                or by using the group option.

            vargs: Additional arguments that get passed into :func:plt.hist.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hist
                for additional arguments that can be passed into vargs. These
                include: `range`, `normed`/`density`, `cumulative`, and
                `orientation`, to name a few.

        >>> t = Table().with_columns(
        ...     'count',  make_array(9, 3, 3, 1),
        ...     'points', make_array(1, 2, 2, 10))
        >>> t
        count | points
        9     | 1
        3     | 2
        3     | 2
        1     | 10
        >>> t.hist() # doctest: +SKIP
        <histogram of values in count>
        <histogram of values in points>

        >>> t = Table().with_columns(
        ...     'value',      make_array(101, 102, 103),
        ...     'proportion', make_array(0.25, 0.5, 0.25))
        >>> t.hist(bin_column='value') # doctest: +SKIP
        <histogram of values weighted by corresponding proportions>

        >>> t = Table().with_columns(
        ...     'value',    make_array(1,   2,   3,   2,   5  ),
        ...     'category', make_array('a', 'a', 'a', 'b', 'b'))
        >>> t.hist('value', group='category') # doctest: +SKIP
        <two overlaid histograms of the data [1, 2, 3] and [2, 5]>
        """
        if counts is not None and bin_column is None:
            warnings.warn("counts arg of hist is deprecated; use bin_column")
            bin_column=counts
        if columns:
            columns_included = list(columns)
            if bin_column is not None:
                columns_included.append(bin_column)
            if group is not None:
                columns_included.append(group)
            self = self.select(*columns_included)
        if group is not None:
            if bin_column is not None:
                raise ValueError("Using bin_column and group together is "
                                 "currently unsupported.")
            if len(columns) > 1:
                raise ValueError("Using group with multiple histogram value "
                                 "columns is currently unsupported.")

        # Check for non-numerical values and raise a ValueError if any found
        for col in self:
            if col != group and any(isinstance(cell, np.flexible) for cell in self[col]):
                raise ValueError("The column '{0}' contains non-numerical "
                    "values. A histogram cannot be drawn for this table."
                    .format(col))


        if bin_column is not None and bins is None:
            bins = np.unique(self.column(bin_column))
        if bins is not None:
            vargs['bins'] = bins

        # Matplotlib has deprecated the normed keyword.
        # TODO consider changing this function to use density= instead too
        if 'normed' not in vargs and 'density' not in vargs:
            vargs['density'] = True
        elif 'normed' in vargs and 'density' not in vargs:
            vargs['density'] = vargs.pop('normed')
        elif 'normed' in vargs and 'density' in vargs:
            raise ValueError("You can't specify both normed and density. "
                             "Use one or the other.")

        def prepare_hist_with_bin_column(bin_column):
            # This code is factored as a function for clarity only.
            weight_columns = [c for c in self.labels if c != bin_column]
            bin_values = self.column(bin_column)
            values_dict = [(w[:-6] if w.endswith(' count') else w, (bin_values, self.column(w))) \
                for w in weight_columns]
            return values_dict

        def prepare_hist_with_group(group):
            # This code is factored as a function for clarity only.
            grouped = self.group(group, np.array)
            if grouped.num_rows > 20:
                warnings.warn("It looks like you're making a grouped histogram with "
                              "a lot of groups ({:d}), which is probably incorrect."
                              .format(grouped.num_rows))
            return [("{}={}".format(group, k), (v[0][1],)) for k, v in grouped.index_by(group).items()]

        # Populate values_dict: An ordered dict from column name to singleton
        # tuple of array of values or a (values, weights) pair of arrays.  If
        # any values have weights, they all must have weights.
        if bin_column is not None:
            values_dict = prepare_hist_with_bin_column(bin_column)
        elif group is not None:
            values_dict = prepare_hist_with_group(group)
        else:
            values_dict = [(k, (self.column(k),)) for k in self.labels]
        values_dict = collections.OrderedDict(values_dict)

        def draw_hist(values_dict):
            with np.printoptions(legacy='1.13'):
                # This code is factored as a function for clarity only.
                n = len(values_dict)
                colors = [rgb_color + (self.default_alpha,) for rgb_color in
                    itertools.islice(itertools.cycle(self.chart_colors), n)]
                hist_names = list(values_dict.keys())
                values = [v[0] for v in values_dict.values()]
                weights = [v[1] for v in values_dict.values() if len(v) > 1]
                if n > len(weights) > 0:
                    raise ValueError("Weights were provided for some columns, but not "
                                     " all, and that's not supported.")
                if vargs['density']:
                    y_label = 'Percent per ' + (unit if unit else 'unit')
                    percentage = plt.FuncFormatter(lambda x, _: "{:g}".format(100*x))
                else:
                    y_label = 'Count'

                if overlay and n > 1:
                    # Reverse because legend prints bottom-to-top
                    values = values[::-1]
                    weights = weights[::-1]
                    colors = list(colors)[::-1]
                    if len(weights) == n:
                        vargs['weights'] = weights
                    if not side_by_side:
                        vargs.setdefault('histtype', 'stepfilled')
                    figure = plt.figure(figsize=(width, height))
                    plt.hist(values, color=colors, **vargs)
                    axis = figure.get_axes()[0]
                    _vertical_x(axis)
                    axis.set_ylabel(y_label)
                    if vargs['density']:
                        axis.yaxis.set_major_formatter(percentage)
                    x_unit = ' (' + unit + ')' if unit else ''
                    if group is not None and len(self.labels) == 2:
                        #There's a grouping in place but we're only plotting one column's values
                        label_not_grouped = [l for l in self.labels if l != group][0]
                        axis.set_xlabel(label_not_grouped + x_unit, fontsize=16)
                    else:    
                        axis.set_xlabel(x_unit, fontsize=16)
                    plt.legend(hist_names, loc=2, bbox_to_anchor=(1.05, 1))
                    type(self).plots.append(axis)
                else:
                    _, axes = plt.subplots(n, 1, figsize=(width, height * n))
                    if 'bins' in vargs:
                        bins = vargs['bins']
                        if isinstance(bins, numbers.Integral) and bins > 76 or hasattr(bins, '__len__') and len(bins) > 76:
                            # Use stepfilled when there are too many bins
                            vargs.setdefault('histtype', 'stepfilled')
                    if n == 1:
                        axes = [axes]
                    for i, (axis, hist_name, values_for_hist, color) in enumerate(zip(axes, hist_names, values, colors)):
                        axis.set_ylabel(y_label)
                        if vargs['density']:
                            axis.yaxis.set_major_formatter(percentage)
                        x_unit = ' (' + unit + ')' if unit else ''
                        if len(weights) == n:
                            vargs['weights'] = weights[i]
                        axis.set_xlabel(hist_name + x_unit, fontsize=16)
                        axis.hist(values_for_hist, color=color, **vargs)
                        _vertical_x(axis)
                        type(self).plots.append(axis)

        draw_hist(values_dict)

    def hist_of_counts(self, *columns, overlay=True, bins=None, bin_column=None,
                       group=None, side_by_side=False, width=6, height=4, **vargs):
        """
        Plots one count-based histogram for each column in columns. The
        heights of each bar will represent the counts, and all the bins
        must be of equal size.

        If no column is specified, plot all columns.

        Kwargs:
            overlay (bool): If True, plots 1 chart with all the histograms
                overlaid on top of each other (instead of the default behavior
                of one histogram for each column in the table). Also adds a
                legend that matches each bar color to its column.  Note that
                if the histograms are not overlaid, they are not forced to the
                same scale.

            bins (array or int): Lower bound for each bin in the
                histogram or number of bins. If None, bins will
                be chosen automatically.

            bin_column (column name or index): A column of bin lower bounds.
                All other columns are treated as counts of these bins.
                If None, each value in each row is assigned a count of 1.

            group (column name or index): A column of categories.  The rows are
                grouped by the values in this column, and a separate histogram is
                generated for each group.  The histograms are overlaid or plotted
                separately depending on the overlay argument.  If None, no such
                grouping is done.

            side_by_side (bool): Whether histogram bins should be plotted side by
                side (instead of directly overlaid).  Makes sense only when
                plotting multiple histograms, either by passing several columns
                or by using the group option.

            vargs: Additional arguments that get passed into :func:plt.hist.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hist
                for additional arguments that can be passed into vargs. These
                include: `range`, `cumulative`, and
                `orientation`, to name a few.

        >>> t = Table().with_columns(
        ...     'count',  make_array(9, 3, 3, 1),
        ...     'points', make_array(1, 2, 2, 10))
        >>> t
        count | points
        9     | 1
        3     | 2
        3     | 2
        1     | 10
        >>> t.hist_of_counts() # doctest: +SKIP
        <histogram of values in count with counts on y-axis>
        <histogram of values in points with counts on y-axis>

        >>> t = Table().with_columns(
        ...     'value', make_array(101, 102, 103),
        ...     'count', make_array(5, 10, 5))
        >>> t.hist_of_counts(bin_column='value') # doctest: +SKIP
        <histogram of values weighted by corresponding counts>

        >>> t = Table().with_columns(
        ...     'value',    make_array(1,   2,   3,   2,   5  ),
        ...     'category', make_array('a', 'a', 'a', 'b', 'b'))
        >>> t.hist('value', group='category') # doctest: +SKIP
        <two overlaid histograms of the data [1, 2, 3] and [2, 5]>
        """

        if bin_column is not None and bins is None:
            bins = np.unique(self.column(bin_column))
            # TODO ensure counts are integers even when `columns` is empty
            for column in columns:
                if not _is_array_integer(self.column(column)):
                    raise ValueError('The column {0} contains non-integer values '
                                     'When using hist_of_counts with bin_columns, '
                                     'all columns should contain counts.'
                                     .format(column))

        if vargs.get('normed', False) or vargs.get('density', False):
            raise ValueError("hist_of_counts is for displaying counts only, "
                             "and should not be used with the normed or "
                             "density keyword arguments")
        vargs['density'] = False

        if bins is not None:
            if len(bins) < 2:
                raise ValueError("bins must have at least two items")
            diffs = np.diff(sorted(bins))
            # Diffs should all be equal (up to floating point error)
            normalized_diff_deviances = np.abs((diffs - diffs[0])/diffs[0])
            if np.any(normalized_diff_deviances > 1e-11):
                raise ValueError("Bins of unequal size should not be used "
                                 "with hist_of_counts. Please use hist() and "
                                 "make sure to set normed=True")
        return self.hist(*columns, overlay=overlay, bins=bins, bin_column=bin_column, group=group, side_by_side=side_by_side, width=width, height=height, **vargs)


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

        >>> table = Table().with_columns(
        ...     'test1', make_array(92.5, 88, 72, 71, 99, 100, 95, 83, 94, 93),
        ...     'test2', make_array(89, 84, 74, 66, 92, 99, 88, 81, 95, 94))
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


def _is_array_integer(arr):
    """Returns True if an array contains integers (integer type or near-int
    float values) and False otherwise.

    >>> _is_array_integer(np.arange(10))
    True
    >>> _is_array_integer(np.arange(7.0, 20.0, 1.0))
    True
    >>> _is_array_integer(np.arange(0, 1, 0.1))
    False
    """
    return issubclass(arr.dtype.type, np.integer) or np.allclose(arr, np.round(arr))

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


def _as_labels(column_or_columns):
    """Return a list of labels for a label or labels."""
    if not _is_non_string_iterable(column_or_columns):
        return [column_or_columns]
    else:
        return column_or_columns

def _varargs_labels_as_list(label_list):
    """Return a list of labels for a list of labels or singleton list of list
    of labels."""
    if len(label_list) == 0:
        return []
    elif not _is_non_string_iterable(label_list[0]):
        # Assume everything is a label.  If not, it'll be caught later.
        return label_list
    elif len(label_list) == 1:
        return label_list[0]
    else:
        raise ValueError("Labels {} contain more than list.".format(label_list),
                         "Pass just one list of labels.")

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

    def __call__(self, row_numbers_or_slice, *args):
        if args:
            all_args = list(args)
            all_args.insert(0, row_numbers_or_slice)
            all_args = np.array(all_args)
        else:
            all_args = row_numbers_or_slice
        return self.__getitem__(all_args)

    @abc.abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError()


class _RowTaker(_RowSelector):
    def __getitem__(self, row_indices_or_slice):
        """Return a new Table with selected rows taken by index.

        Args:
            ``row_indices_or_slice`` (integer or array of integers):
            The row index, list of row indices or a slice of row indices to
            be selected.

        Returns:
            A new instance of ``Table`` with selected rows in order
            corresponding to ``row_indices_or_slice``.

        Raises:
            ``IndexError``, if any ``row_indices_or_slice`` is out of bounds
            with respect to column length.

        >>> grades = Table().with_columns('letter grade',
        ...     make_array('A+', 'A', 'A-', 'B+', 'B', 'B-'),
        ...     'gpa', make_array(4, 4, 3.7, 3.3, 3, 2.7))
        >>> grades
        letter grade | gpa
        A+           | 4
        A            | 4
        A-           | 3.7
        B+           | 3.3
        B            | 3
        B-           | 2.7
        >>> grades.take(0)
        letter grade | gpa
        A+           | 4
        >>> grades.take(-1)
        letter grade | gpa
        B-           | 2.7
        >>> grades.take(make_array(2, 1, 0))
        letter grade | gpa
        A-           | 3.7
        A            | 4
        A+           | 4
        >>> grades.take[:3]
        letter grade | gpa
        A+           | 4
        A            | 4
        A-           | 3.7
        >>> grades.take(np.arange(0,3))
        letter grade | gpa
        A+           | 4
        A            | 4
        A-           | 3.7
        >>> grades.take(0, 2)
        letter grade | gpa
        A+           | 4
        A-           | 3.7
        >>> grades.take(10)
        Traceback (most recent call last):
            ...
        IndexError: index 10 is out of bounds for axis 0 with size 6
        """
        if isinstance(row_indices_or_slice, collections.abc.Iterable):
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

        >>> t = Table().with_columns(
        ...     'letter grade', make_array('A+', 'A', 'A-', 'B+', 'B', 'B-'),
        ...     'gpa', make_array(4, 4, 3.7, 3.3, 3, 2.7))
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
        >>> t.exclude(make_array(1, 3, 4))
        letter grade | gpa
        A+           | 4
        A-           | 3.7
        B-           | 2.7
        >>> t.exclude(range(3))
        letter grade | gpa
        B+           | 3.3
        B            | 3
        B-           | 2.7
        >>> t.exclude(0, 2)
        letter grade | gpa
        A            | 4
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
        if isinstance(row_indices_or_slice, collections.abc.Iterable):
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
