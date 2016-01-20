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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import IPython

import datascience.maps as _maps
import datascience.formats as _formats
import datascience.util as _util


class Table(collections.abc.MutableMapping):
    """A sequence of string-labeled columns."""

    def __init__(self, labels=[], formatter=_formats.default_formatter):
        """Create an empty table with column labels.

        >>> tiles = Table(['letter', 'count', 'points'])
        >>> tiles
        letter | count | points

        Args:
            ``labels`` (list of strings): The column labels.

            ``formatter`` (Formatter): An instance of :class:`Formatter` that
                formats the columns' values.
        """
        for label in labels:
            self._check_label(label)
        self.df = pd.DataFrame(columns=labels)
        self._formats = dict() # TODO formats
        self.formatter = formatter

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
        except AttributeError:
            path = filepath_or_buffer
        try:
            if 'sep' not in vargs and path.endswith('.csv'):
                vargs['sep'] = ','
        except AttributeError:
            pass

        df = pandas.read_table(filepath_or_buffer, *args, **vargs)
        return cls.from_df(df)

    @classmethod
    def from_df(cls, df):
        """Return a table with the contents of a DataFrame."""
        assert isinstance(df, pd.DataFrame)
        t = Table()
        t.df = df
        return t

    @classmethod
    def from_columns(cls, columns):
        """Return a table with additional or replaced columns.

        Args:
            ``columns`` (list): An alternating sequence of labels and values.

        >>> Table().with_columns([
        ...     'letter', ['c', 'd'],
        ...     'count', [3, 2],
        ... ])
        letter | count
        c      | 3
        d      | 2
        """
        assert len(columns) % 2 == 0, 'columns must alternate labels and values'
        t = cls()
        for i in range(len(columns), 2):
            label, value = columns[i], columns[i+1]
            assert cls._check_label(columns[i])
            t.df[label] = value
        return t

    @staticmethod
    def _check_label(label):
        if not isinstance(label, str):
            raise ValueError('Column label not a string: ' + str(type(label)))

    ##################################
    # Non-mutating addition/deletion #
    ##################################

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
        if len(row) != self.num_columns:
            raise ValueError('Row length mismatch')
        other = pd.Series(row, index=self.df.columns)
        return self.from_df(self.df.append(other, ignore_index=True))

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
        if isinstance(rows, np.ndarray):
            rows = rows.tolist()
        for row in rows:
            if len(row) != self.num_columns:
                raise ValueError('Row length mismatch')
        other = pd.DataFrame.from_records(rows)
        other.columns = self.df.columns
        return self.from_df(self.df.append(other, ignore_index=True))

    def with_column(self, label, value_or_values):
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
        ...     'count', [2, 4],
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
        self._check_label(label)
        df = self.df.copy()
        df[label] = value_or_values
        return self.from_df(df)

    def with_columns(self, labels_and_values):
        """Return a table with additional or replaced columns.

        Args:
            ``labels_and_values``: An alternating list of labels and values.


        >>> tiles = Table().with_columns([
        ...     'letter', ['c', 'd'],
        ...     'count', [2, 4],
        ... ])
        >>> tiles
        letter | count
        c      | 2
        d      | 4
        """
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

    def _with_columns(self, columns):
        """Create a table from a sequence of columns, copying column labels."""
        table = Table()
        for label, column in zip(self.labels, columns):
            self._set_column_and_format(table, label, column)
        return table

    ######################################################################
    # Magic Methods: A Table is a mutable mapping from labels to columns #
    ######################################################################

    def __getitem__(self, index_or_label):
        return self.column(index_or_label)

    def __setitem__(self, index_or_label, values):
        self.set_column(index_or_label, values)

    def __delitem__(self, index_or_label):
        self.remove_column(index_or_label)

    def __len__(self):
        return self.num_columns

    def __iter__(self):
        return iter(self.labels)

    ######################################
    # Expose DataFrame/Series attributes #
    ######################################

    # TODO be comprehensive
    _dataframe_attrs = ['cumsum']
    _series_attrs = []

    def __getattr__(self, attr):
        """Return a method from the underlying dataframe."""
        if attr in self._dataframe_attrs and hasattr(self.df, attr):
            result = getattr(self.df, attr)
            if callable(result):
                @functools.wraps(result)
                def method(*args, **vargs):
                    """Create a table when a dataframe is returned."""
                    value = result(*args, **vargs)
                    if isinstance(value, pd.DataFrame):
                        value = self.from_df(value)
                    return value
                return method
            else:
                return result
        elif attr in self._series_attrs and self.labels and all(hasattr(self[s], attr) for s in self.labels):
            attrs = [getattr(c, attr) for c in self.columns]
            return _with_columns_wrapped(attrs, self)
        else:
            msg = "'{0}' object has no attribute '{1}'".format(type(self).__name__, attr)
            raise AttributeError(msg)

    @property
    def str(self):
        return _Methods(self, 'str')

    ####################
    # Accessing Values #
    ####################

    @property
    def num_rows(self):
        """Number of rows."""
        return self.df.shape[0]

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
        return tuple(self.df.columns)

    @property
    def num_columns(self):
        """Number of columns."""
        return len(self.labels)

    @property
    def columns(self):
        """Return a sequence of all columns."""
        return [self.column(i) for i in range(self.num_columns)]

    def column(self, index_or_label):
        """Return the values of a column as a 0-indexed series.

        table.column(label) is equivalent to table[label].

        Args:
            label (int or str): The index or label of a column

        Returns:
            A ``pd.Series`` instance.
        """
        label = self._as_label(index_or_label)
        return self.df[label]

    def column_index(self, label):
        """Return the index of a column."""
        return self.labels.index(label)

    def apply(self, fn, label_or_labels):
        """Returns a series where fn is applied to each set of elements
        by row from the specified columns in label_or_labels.

        Args:
            ``fn`` (function): An n-ary function, where ``n`` is the number of
                labels selected.

            ``label`` (str or sequences of str): column labels selecting data
                to be passed to ``fn``.

        Raises:
            ``ValueError``: ``label`` does not appear in the table.

        Returns:
            A series of results of applying ``fn`` to elements specified by
            ``label`` in each row.

        >>> tiles = Table(['letter', 'count'])
        >>> tiles = tiles.with_rows([['c', 2], ['d', 4]])
        >>> tiles.apply(lambda x, y: x * int(y), ['letter', 'count'])
        0      cc
        1    dddd
        dtype: object
        >>> tiles.apply(lambda x: x + 1, 'count')
        0    3
        1    5
        dtype: float64
        """
        return self.select(label_or_labels).df.apply(lambda s: fn(*s), axis=1)

    ############
    # Mutation #
    ############

    def set_format(self, label_or_labels, formatter):
        """Set the format of a column."""
        # TODO
        if inspect.isclass(formatter) and issubclass(formatter, _formats.Formatter):
            formatter = formatter()
        for label in self._as_labels(label_or_labels):
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

    def move_to_start(self, label):
        """Move a column to the first in order."""
        # TODO
        self._columns.move_to_end(label, last=False)
        return self

    def move_to_end(self, label):
        """Move a column to the last in order."""
        # TODO
        self._columns.move_to_end(label)
        return self

    def append(self, row_or_table):
        """Append a row or all rows of a table. An appended table must have all
        columns of self."""
        # TODO
        if not row_or_table:
            return
        if isinstance(row_or_table, Table):
            t = row_or_table
            row = list(t.select(self.labels)._columns.values())
            n = t.num_rows
        else:
            row, n = row_or_table, 1
        for i, column in enumerate(self._columns):
            self._columns[column] = np.append(self[column], row[i])
        self._num_rows += n
        return self

    def set_column(self, label, value_or_values):
        """Add or update a column of the table.

        ``table.set_column('x', [1, 2, 3])`` is equivalent to
        ``table['x'] = [1, 2, 3]``.

        Args:
            ``label`` (str): The label of the new column.

            ``value_or_values`` (single value or sequence): If a single value,
                every value in the new column is ``values``.

                If a sequence, the new column contains the values in
                ``values``. ``values`` must be the same length as the table.

        Returns:
            Original table, modified to contain the new/updated column

        Raises:
            ``ValueError``: If
                - ``label`` is not a string.
                - ``value_or_values`` is a list/array and does not have the
                  same length as the number of rows in the table.

        >>> tiles = Table(['letter', 'count'])
        >>> tiles = tiles.with_rows([['c', 2], ['d', 4]])
        >>> tiles.add_column('points', [3, 2])
        letter | count | points
        c      | 2     | 3
        d      | 4     | 2
        >>> tiles.add_column('count', 1)
        letter | count | points
        c      | 1     | 3
        d      | 1     | 2
        """
        # TODO(sam): Allow set_column to take in a another table, copying
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

        self.df[label] = values

    def relabel(self, label, new_label):
        """Change the labels of columns specified by ``label`` to
        labels in ``new_label``.

        Args:
            ``label`` (str or sequence of str): The label(s) of
                columns to be changed.

            ``new_label`` (str or sequence of str): The new label(s) of
                columns to be changed.

                Number of elements must match number of elements in
                ``label``.

        Returns:
            Original table with modified labels

        >>> tiles = Table(['letter', 'count'])
        >>> tiles = tiles.with_rows([['c', 2], ['d', 4]])
        >>> tiles.relabel('count', 'number')
        letter | number
        c      | 2
        d      | 4
        >>> tiles
        letter | number
        c      | 2
        d      | 4
        """
        if isinstance(label, str) and isinstance(new_label, str):
            label, new_label = [label], [new_label]
        if len(label) != len(new_label):
            raise ValueError('Invalid arguments. label and new_label '
                'must be of equal length.')
        for old, new in zip(label, new_label):
            if not (old in self.labels):
                raise ValueError(old + ' is not an existing label')
            if (new in self.labels):
                raise ValueError(new + ' is an existing label')
        for old, new in zip(label, new_label):
            self.df.columns[self.column_index(old)] = new

    def _set_column_and_format(self, table, label, column):
        """Add a column to table, copying the formatter from self."""
        assert isinstance(label, str)
        table.df[label] = column
        if label in self._formats: # TODO
            table._formats[label] = self._formats[label]

    ##################
    # Transformation #
    ##################

    def copy(self):
        """Return a copy of a Table."""
        return self.from_df(self.df.copy())

    def select(self, label_or_labels):
        """Return a Table with selected column or columns by label.

        Args:

            ``label_or_labels`` (str or index or strs or indices): The
            columns to be selected.

        Returns:
            An instance of ``Table`` containing only selected columns.

        >>> t = Table().with_columns([
        ...     'burgers', ['cheeseburger', 'hamburger', 'veggie'],
        ...     'prices',  [6, 5, 5],
        ...     'calories', [743, 651, 582],
        ... ])
        >>> t
        burgers      | prices | calories
        cheeseburger | 6      | 743
        hamburger    | 5      | 651
        veggie       | 5      | 582
        >>> t.select(['burgers', 'calories'])
        burgers      | calories
        cheeseburger | 743
        hamburger    | 651
        veggie       | 582
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
        """
        labels = self._as_labels(label_or_labels)
        table = Table()
        for label in labels:
            self._set_column_and_format(table, label, np.copy(self[label]))
        return table

    def take(self, rows):
        """Return a Table with specified row numbers."""
        t = self.df.iloc[rows]
        t = t.reset_index(drop=True)
        return self.from_df(t)

    def exclude(self):
        raise NotImplementedError()

    def drop(self, label_or_labels):
        """Return a Table with only columns other than selected label or labels."""
        exclude = self._as_labels(label_or_labels)
        return self.select([c for c in self.labels if c not in exclude])

    def where(self, column_or_label, value=None):
        """Return a Table of rows for which the column is value or a non-zero value."""
        column = self._get_column(column_or_label)
        if value is not None:
            column = column == value
        return self.take(np.nonzero(column)[0])

    def sort(self, column_or_label, descending=False, distinct=False):
        """Return a Table of sorted rows by the values in a column."""
        column = self._get_column(column_or_label)
        if distinct:
            _, row_numbers = np.unique(column, return_index=True)
        else:
            row_numbers = np.argsort(column, axis=0, kind='mergesort')
        assert (row_numbers < self.num_rows).all(), row_numbers
        if descending:
            row_numbers = np.array(row_numbers[::-1])
        return self.take(row_numbers)

    def group(self, column_or_label, collect=lambda s: s):
        """Group rows by unique values in label, aggregating values.

        collect -- an optional function applied to the values for each group.

        The grouped column will appear first in the result table.
        """
        self = self._with_columns(self.columns)  # Shallow self
        collect = _zero_on_type_error(collect)

        # Remove column used for grouping
        column = self._get_column(column_or_label)
        if column_or_label in self.labels:
            label = column_or_label
            del self[label]
        else:
            label = self._unused_label('group')

        # Generate grouped columns
        groups = self.index_by(column)
        keys = sorted(groups.keys())
        columns, labels = [], []
        for i, label in enumerate(self.labels):
            labels.append(_collected_label(collect, label))
            c = [collect(np.array([row[i] for row in groups[k]])) for k in keys]
            columns.append(c)

        grouped = type(self)(columns, labels)
        assert label == self._unused_label(label)
        grouped[label] = keys
        grouped.move_to_start(label)
        return grouped

    def groups(self, labels, collect=lambda s: s):
        """Group rows by multiple columns, aggregating values."""
        collect = _zero_on_type_error(collect)
        columns = []
        for label in labels:
            assert label in self.labels
            columns.append(self._get_column(label))
        grouped = self.group(list(zip(*columns)))
        grouped._columns.popitem(last=False) # Discard the column of tuples

        # Flatten grouping values and move them to front
        for label in labels[::-1]:
            grouped[label] = grouped.apply(_assert_same, label)
            grouped.move_to_start(label)

        # Aggregate other values
        for label in grouped.labels:
            if label in labels:
                continue
            column = [collect(v) for v in grouped[label]]
            del grouped[label]
            grouped[_collected_label(collect, label)] = column
        return grouped

    def pivot(self, columns, rows, values, collect=lambda s:s, zero=None):
        """Generate a table with a column for rows (or a column for each row
        in rows list) and a column for each unique value in columns. Each row
        aggregates over the values that match both row and column.

        columns, values -- column labels in self
        rows -- column label or a list of column labels
        collect -- aggregation function over values
        zero -- zero value for non-existent row-column combinations
        """
        rows = self._as_labels(rows)
        selected = self.select([columns, values] + rows)
        grouped = selected.groups([columns] + rows, collect)

        # Generate existing combinations of values from columns in rows
        rows_values = sorted(list(set(self.select(rows).rows)))
        pivoted = Table.from_rows(rows_values, rows)

        # Generate other columns and add them to pivoted
        by_columns = grouped.index_by(columns)
        for label in sorted(by_columns):
            tuples = [t[1:] for t in by_columns[label]] # Discard column value
            column = _fill_with_zeros(rows_values, tuples, zero)
            pivot = self._unused_label(str(label) + ' ' + values)
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
        pivot_columns = self._as_labels(pivot_columns)
        selected = self.select(pivot_columns + [value_column])
        grouped=selected.groups(pivot_columns)

        # refine bins by taking a histogram over all the data
        if bins is not None:
            vargs['bins'] = bins
        _, rbins = np.histogram(self[value_column],**vargs)
        # create a table with these bins a first column and counts for each group
        vargs['bins'] = rbins
        binned = Table([rbins],['bin'])
        for group in grouped.rows:
            col_label = "-".join(map(str,group[0:-1]))
            col_vals = group[-1]
            counts,_ = np.histogram(col_vals,**vargs)
            binned[col_label] = np.append(counts,0)
        return binned

    def stack(self, key, labels=None):
        """
        Takes k original columns and returns two columns, with col. 1 of
        all column names and col. 2 of all associated data.
        """
        rows, labels = [], labels or self.labels
        for row in self.rows:
            [rows.append((getattr(row, key), k, v)) for k, v in row.asdict().items()
             if k != key and k in labels]
        return Table.from_rows(rows, [key, 'column', 'value'])

    def join(self, label, other, other_label=None):
        """Generate a table with the columns of self and other, containing rows
        for all values of a column that appear in both tables.
        If a join value appears more than once in self, each row will be used,
        but in the other table, only the first of each will be used.

        If the result is empty, return None.
        """
        if self.num_rows == 0 or other.num_rows == 0:
            return None
        if not other_label:
            other_label = label

        self_rows = self.index_by(label)
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
        joined = Table.from_rows(joined_rows, labels)
        del joined[self._unused_label(other_label)] # Remove redundant column
        return joined.move_to_start(label).sort(label)

    def stats(self, ops=(min, max, np.median, sum)):
        """Compute statistics for each column and place them in a table."""
        names = [op.__name__ for op in ops]
        ops = [_zero_on_type_error(op) for op in ops]
        columns = [[op(column) for op in ops] for column in self.columns]
        table = Table(columns, self.labels)
        stats = table._unused_label('statistic')
        table[stats] = names
        table.move_to_start(stats)
        return table

    def _unused_label(self, label):
        """Generate an unused label."""
        original = label
        existing = self.labels
        i = 2
        while label in existing:
            label = '{}_{}'.format(original, i)
            i += 1
        return label

    def _as_label(self, index_or_label):
        if isinstance(index_or_label, str):
            return index_or_label
        if isinstance(index_or_label, numbers.Integral):
            return self.labels[index_or_label]
        else:
            raise ValueError(str(index_or_label) + ' is not a label or index')

    def _as_labels(self, label_or_labels):
        """Return a list of labels for one or more labels or column indices."""
        if not _is_non_string_iterable(label_or_labels):
            items = [label_or_labels]
        else:
            items = label_or_labels
        return [self._as_label(s) for s in items]

    def _get_column(self, column_or_label):
        """Convert label to column and check column length."""
        c = column_or_label
        if isinstance(c, collections.Hashable) and c in self.labels:
            return self.column(c).values
        if isinstance(c, numbers.Integral):
            return self.columns[c].values
        elif isinstance(c, str):
            assert c in self.labels, 'label "{}" not in labels {}'.format(c, self.labels)
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

        >>> count = [9, 3, 3, 1]
        >>> points = [1, 2, 2, 10]
        >>> table = Table([count, points], ['count', 'points'])
        >>> table
        count | points
        9     | 1
        3     | 2
        3     | 2
        1     | 10
        >>> table.percentile(67)
        count | points
        9     | 10
        """
        percentiles = [[_util.percentile(p, self[column_name])]
                       for column_name in self]
        return Table(percentiles, self.labels)

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

        >>> job = ['a', 'b', 'c', 'd']
        >>> wage = [10, 20, 15, 8]
        >>> some_table = Table([job, wage], ['job', 'wage'])
        >>> some_table
        job  | wage
        a    | 10
        b    | 20
        c    | 15
        d    | 8
        >>> some_table.sample() # doctest: +SKIP
        job  | wage
        b    | 20
        c    | 15
        a    | 10
        d    | 8
        >>> some_table.sample(k = 2) # doctest: +SKIP
        job  | wage
        b    | 20
        c    | 15
        >>> some_table.sample(k = 2, with_replacement = True,
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
        sample = Table(columns, self.labels)
        for label in self._formats:
            sample._formats[label] = self._formats[label]
        return sample

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

        >>> job = ['a', 'b', 'c', 'd']
        >>> wage = [10, 20, 15, 8]
        >>> foo_table = Table([job, wage], ['job', 'wage'])
        >>> foo_table
        job  | wage
        a    | 10
        b    | 20
        c    | 15
        d    | 8
        >>> sample, rest = foo_table.split(3)
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
        first = Table.from_rows(rows[:k], self.labels)
        rest = Table.from_rows(rows[k:], self.labels)
        for label in self._formats :
            first._formats[label] = self._formats[label]
            rest._formats[label] = self._formats[label]
        return first, rest


    def bin(self, **vargs):
        """Group values by bin and compute counts per bin by column.

        By default, bins are chosen to contain all values in all columns. The
        following named arguments from numpy.histogram can be applied to
        specialize bin widths:

        If the original table has n columns, the resulting binned table has
        n+1 columns, where column 0 contains the lower bound of each bin.

        Args:
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
        if 'normed' in vargs:
            vargs.setdefault('density', vargs.pop('normed'))
        density = vargs.get('density', False)
        tag = 'density' if density else 'count'

        cols = list(self._columns.values())
        _, bins = np.histogram(cols, **vargs)

        binned = Table([bins], ['bin'])
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
        labels = self.labels
        fmts = [self._formats.get(k, self.formatter.format_column(k, self.column(k)[:max_rows])) for
            k in self.labels]
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
        fmts = [self._formats.get(k, self.formatter.format_column(k, self.column(k)[:max_rows])) for
            k in self.labels]
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

    def to_csv(self, filename):
        """Creates a CSV file with the provided filename.

        The CSV is created in such a way that if we run
        ``table.to_csv('my_table.csv')`` we can recreate the same table with
        ``Table.read_table('my_table.csv')``.

        Args:
            ``filename`` (str): The filename of the output CSV file.

        Returns:
            None, outputs a file with name ``filename``.

        >>> job = ['a', 'b', 'c', 'd']
        >>> wage = [10, 20, 15, 8]
        >>> some_table = Table([job, wage], ['job', 'wage'])
        >>> some_table
        job  | wage
        a    | 10
        b    | 20
        c    | 15
        d    | 8
        >>> some_table.to_csv('my_table.csv') # doctest: +SKIP
        <outputs a file called my_table.csv in the current directory>
        """
        # We use index = False to avoid the row number output that pandas does
        # by default.
        self.todf().to_csv(filename, index = False)

    def to_array(self):
        """Convert the table to a NumPy array."""
        dt = np.dtype(list(zip(self.labels,
                               (c.dtype for c in self.columns))))
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
        'alpha': 0.8,
    }

    def _visualize(self, x_label, y_labels, ticks, overlay, draw, annotate, width=6, height=4):
        """Generic visualization that overlays or separates the draw function."""
        n = len(y_labels)
        colors = list(itertools.islice(itertools.cycle(self.chart_colors), n))
        if overlay:
            _, axis = plt.subplots(figsize=(width, height))
            for label, color in zip(y_labels, colors):
                draw(axis, label, color)
            if ticks is not None:
                annotate(axis, ticks)
            axis.legend(y_labels, bbox_to_anchor=(1.5, 1.0))
        else:
            fig, axes = plt.subplots(n, 1, figsize=(width, height * n))
            if not isinstance(axes, collections.Iterable):
                axes=[axes]
            for axis, y_label, color in zip(axes, y_labels, colors):
                draw(axis, y_label, color)
                axis.set_ylabel(y_label, fontsize=16)
                axis.set_xlabel(x_label, fontsize=16)
                if ticks is not None:
                    annotate(axis, ticks)

    def plot(self, column_for_xticks=None, overlay=False, **vargs):
        """Plot contents as lines."""
        options = self.default_options.copy()
        options.update(vargs)

        xticks, y_labels = self._split_by_column(column_for_xticks)

        def draw(axis, label, color):
            if xticks is None:
                axis.plot(self[label], color=color, **options)
            else:
                axis.plot(xticks, self[label], color=color, **options)

        def annotate(axis, ticks):
            axis.set_xticklabels(axis.get_xticks(), rotation='vertical')

        self._visualize(column_for_xticks, y_labels, xticks, overlay, draw, annotate)

    def bar(self, column_for_categories=None, overlay=False, **vargs):
        """Plots bar charts for the table.

        Each chart is categorized using the values in `column_for_categories`
        and one chart is produced for every other column in the table.
        A total of n - 1 charts are created where n is the number of columns
        in the table.

        Requires every column except for `column_for_categories` to be
        numerical. If the columns contain other types, a `ValueError` is
        raised.

        Args:
            column_for_categories (str): The name to use for the bar chart
                categories

        Kwargs:
            overlay (bool): If True, creates one chart with n - 1 bars for each
                category, one for each column other than `column_for_categories`
                (instead of the default behavior of creating n - 1 charts).
                Also adds a legend that matches each bar color to its column.

            vargs: Additional arguments that get passed into `plt.bar`.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.bar
                for additional arguments that can be passed into vargs. These
                include: `linewidth`, `xerr`, `yerr`, and `log`, to name a few.

        Returns:
            None

        Raises:
            ValueError: The Table contains non-numerical values in columns
            other than `column_for_categories`

        """
        options = self.default_options.copy()
        options.update(vargs)

        xticks, y_labels = self._split_by_column(column_for_categories)
        for label in y_labels:
            if any(isinstance(cell, np.flexible) for cell in self[label]):
                raise ValueError("The column '{0}' contains non-numerical "
                    "values. A bar graph cannot be drawn for this table."
                    .format(label))

        index = np.arange(self.num_rows)
        margin = 0.1
        width = 1 - 2 * margin
        if overlay:
            width /= len(y_labels)

        def draw(axis, label, color):
            if overlay:
                xpos = index + margin + (1-2*margin)*labels.index(label)/len(labels)
            else:
                xpos = index
            axis.bar(xpos, self[label], 1.0, color=color, **options)

        def annotate(axis, ticks):
            if (ticks is not None) :
                tick_labels = [ticks[int(l)] for l in axis.get_xticks() if l<len(ticks)]
                axis.set_xticklabels(tick_labels, stretch='ultra-condensed')
            return None
        self._visualize(column_for_categories, y_labels, xticks, overlay, draw, annotate)

    def barh(self, column_for_categories, overlay=False, **vargs):
        """Plots horizontal bar charts for the table.

        Each chart is categorized using the values in `column_for_categories`
        and one chart is produced for every other column in the table.
        A total of n - 1 charts are created where n is the number of columns
        in the table.

        Note that the order of the categories in the table is not guaranteed to
        be preserved in the bar graph. Ex. `barh` on a table with "a", "b", "c"
        as the rows in the `column_for_categories` may not output a bar graph
        with the labels in that order.

        Requires every column except for `column_for_categories` to be
        numerical. If the columns contain other types, a `ValueError` is
        raised.

        Args:
            column_for_categories (str): The name to use for the bar chart
                categories

        Kwargs:
            overlay (bool): If True, creates one chart with n - 1 bars for each
                category, one for each column other than `column_for_categories`
                (instead of the default behavior of creating n - 1 charts).
                Also adds a legend that matches each bar color to its column.

            vargs: Additional arguments that get passed into `plt.barh`.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.barh
                for additional arguments that can be passed into vargs. These
                include: `linewidth`, `xerr`, `yerr`, and `log`, to name a few.

        Returns:
            None

        Raises:
            ValueError: The Table contains non-numerical values in columns
            other than `column_for_categories`

        >>> furniture_table = Table().with_columns([
        ...     'Type of furniture', ['chairs', 'tables', 'desks'],
        ...     'Count', [6, 1, 2]])
        >>> furniture_table
        Type of furniture | Count
        chairs            | 6
        tables            | 1
        desks             | 2
        >>> furniture_table.barh('Type of furniture') # doctest: +SKIP
        <bar graph with chairs, tables, desks as the categories and bars of
        length 6, 1, 2, respectively>
        >>> furniture_table.barh('Count')
        Traceback (most recent call last):
            ...
        ValueError: The column 'Type of furniture' contains non-numerical values. A bar graph cannot be drawn for this table.
        """
        options = self.default_options.copy()
        options.update(vargs)

        yticks, y_labels = self._split_by_column(column_for_categories)
        for label in y_labels:
            if any(isinstance(cell, np.flexible) for cell in self[label]):
                raise ValueError("The column '{0}' contains non-numerical "
                    "values. A bar graph cannot be drawn for this table."
                    .format(label))

        index = np.arange(self.num_rows)
        margin = 0.1
        width = 1 - 2 * margin
        if overlay:
            width /= len(y_labels)

        def draw(axis, label, color):
            if overlay:
                ypos = index + margin + (1-2*margin)*labels.index(label)/len(labels)
            else:
                ypos = index
            # barh plots entries in reverse order from bottom to top
            axis.barh(ypos, self[label][::-1], width,  color=color, **options)

        def annotate(axis, ticks):
            axis.set_yticks(index+0.5) # Center labels on bars
            # barh plots entries in reverse order from bottom to top
            axis.set_yticklabels(ticks[::-1], stretch='ultra-condensed')
        height = max(4, len(index)/2)
        if 'height' in vargs:
            height = vargs.pop('height')
        self._visualize(column_for_categories, y_labels, yticks, overlay, draw, annotate, height=height)

    def _split_by_column(self, column_or_label):
        """Return the specified column and labels of other columns."""
        labels = list(self.labels)
        if column_or_label is None:
            return None, labels
        if column_or_label in labels:
            labels.remove(column_or_label)
        column = self._get_column(column_or_label)
        return column, labels

    def pivot_hist(self, pivot_label, value_label, overlay=False, **vargs):
        """Draw histograms of each category in a column."""
        pvt_labels = np.unique(self[pivot_label])
        pvt_columns = [self[value_label][np.where(self[pivot_label] == pivot)] for pivot in pvt_labels]
        n = len(pvt_labels)
        colors = list(itertools.islice(itertools.cycle(('b', 'g', 'r')), n))
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
                if not isinstance(bins, np.ndarray): bins = abins
                else: assert bins.all() == abins.all(), "Inconsistent bins in hist"
        t = Table()
        t['start'] = bins[0:-1]
        t['end'] = bins[1:]
        for label, column in zip(pvt_labels,vals):
            t[label] = column
        return t

    def hist(self, overlay=False, bins=None, counts=None, **vargs):
        """Plots one histogram for each column in the table.

        Requires all columns in the table to contain numerical values only.
        If the columns contain other types, a ValueError is raised.

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

        Returns:
            None

        Raises:
            ValueError: The Table contains non-numerical values

        >>> count = [9, 3, 3, 1]
        >>> points = [1, 2, 2, 10]
        >>> table = Table([count, points], ['count', 'points'])
        >>> table
        count | points
        9     | 1
        3     | 2
        3     | 2
        1     | 10
        >>> table.hist() # doctest: +SKIP
        <histogram of values in count>
        <histogram of values in points>

        >>> value = [101, 102, 103]
        >>> prop = [0.25, 0.5, 0.25]
        >>> t = Table([value, prop], ['value', 'proportion'])
        >>> t.hist(counts='value') # doctest: +SKIP
        <histogram of values in prop weighted by corresponding values in value>
        """
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

        counted_values = counted_label = None
        if counts is not None:
            counted_values = self._get_column(counts)
            counted_label = 'counts'
            if isinstance(counts, collections.Hashable) and counts in self.labels:
                columns.pop(counts)
                counted_label = counts

        n = len(columns)
        colors = [rgb_color + (self.default_hist_alpha,) for rgb_color in
            itertools.islice(itertools.cycle(self.chart_colors), n)]
        if overlay:
            if counted_values is None:
                values = list(columns.values())[::-1] # Reverse to match legend
            else:
                values = np.repeat(counted_values, n).reshape(-1,n)
                vargs['weights'] = list(columns.values())[::-1] # Reverse to match legend
            vargs.setdefault('histtype', 'stepfilled')
            plt.figure(figsize=(6, 4))
            plt.hist(values, color=colors, **vargs)
            plt.legend(columns.keys())
        else:
            _, axes = plt.subplots(n, 1, figsize=(6, 4 * n))
            if n == 1:
                axes = [axes]
            for axis, label, color in zip(axes, columns.keys(), colors):
                if counted_values is None:
                    values = columns[label]
                    axis.set_xlabel(label, fontsize=16)
                else:
                    values = counted_values
                    axis.set_xlabel(counted_label, fontsize=16)
                    vargs['weights'] = columns[label]
                axis.hist(values, color=color, **vargs)

    def boxplot(self, **vargs):
        """Plots a boxplot for the table.

        Kwargs:
            vargs: Additional arguments that get passed into `plt.boxplot`.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.boxplot
                for additional arguments that can be passed into vargs. These include
                `vert` and `showmeans`.

        Returns:
            None

        Raises:
            ValueError: The Table contains columns with non-numerical values.

        >>> test1_scores = [92.5, 88, 72, 71, 99, 100, 95, 83, 94, 93]
        >>> test2_scores = [89, 84, 74, 66, 92, 99, 88, 81, 95, 94]
        >>> table = Table([test1_scores, test2_scores], ['test1', 'test2'])
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

    def points(self, column__lat, column__long, labels=None, colors=None, **kwargs) :
        latitudes = self._get_column(column__lat)
        longitudes = self._get_column(column__long)
        if labels is not None : labels = self._get_column(labels)
        if colors is not None : colors = self._get_column(colors)
        return _maps.Circle.map(latitudes, longitudes, labels=labels, colors=colors, **kwargs)

    def scatter(self, column_for_x, overlay=False, fit_line=False, **vargs):
        """Creates scatterplots, optionally adding a line of best fit.

        All scatterplots use the values in ``column_for_x`` as the x-values. A
        total of n - 1 scatterplots are created where n is the number of
        columns in the table, one for every column other than ``column_for_x``.

        Requires all columns in the table to contain numerical values only.
        If the columns contain other types, a ``ValueError`` is raised.

        Args:
            ``column_for_x`` (str): The name to use for the x-axis values of the
                scatter plots.

        Kwargs:
            ``overlay`` (bool): If True, creates one scatterplot with n - 1
                y-values plotted, one for each column other than
                ``column_for_x`` (instead of the default behavior of creating n
                - 1 scatterplots. Also adds a legend that matches each dot
                and best-fit line color to its column.

            ``fit_line`` (bool): If True, draws a line of best fit for each
                scatterplot drawn.

            ``vargs``: Additional arguments that get passed into `plt.scatter`.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter
                for additional arguments that can be passed into vargs. These
                include: `marker` and `norm`, to name a couple.

        Returns:
            None

        Raises:
            ``ValueError``: The table contains non-numerical values in columns.

        >>> x = [9, 3, 3, 1]
        >>> y = [1, 2, 2, 10]
        >>> z = [3, 4, 5, 6]
        >>> table = Table([x, y, z], ['x', 'y', 'z'])
        >>> table
        x    | y    | z
        9    | 1    | 3
        3    | 2    | 4
        3    | 2    | 5
        1    | 10   | 6
        >>> table.scatter('x') # doctest: +SKIP
        <scatterplot of values in y on x>
        <scatterplot of values in z on x>

        >>> table.scatter('x', overlay = True) # doctest: +SKIP
        <scatterplot of values in y and z on x>

        >>> table.scatter('x', fit_line = True) # doctest: +SKIP
        <scatterplot of values in y on x with line of best fit>
        <scatterplot of values in z on x with line of best fit>

        """
        # Check for non-numerical values and raise a ValueError if any found
        for col in self:
            if any(isinstance(cell, np.flexible) for cell in self[col]):
                raise ValueError("The column '{0}' contains non-numerical "
                    "values. A histogram cannot be drawn for this table."
                    .format(col))

        options = self.default_options.copy()
        options.update(vargs)
        xdata, y_labels =  self._split_by_column(column_for_x)

        def draw(axis, label, color):
            axis.scatter(xdata, self[label], color=color, **options)
            if fit_line:
                m,b = np.polyfit(xdata, self[label], 1)
                minx, maxx = np.min(xdata),np.max(xdata)
                axis.plot([minx,maxx],[m*minx+b,m*maxx+b])

        def annotate(axis, ticks):
            return None
        self._visualize(column_for_x, y_labels, None, overlay, draw, annotate)

    ###########
    # Support #
    ###########

    class Row(tuple):
        _table = None  # Set by subclasses in Rows

        def __getattr__(self, label):
            return self[self._table.column_index(label)]

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
            return self._row(c[i] for c in self._table.columns)

        def __len__(self):
            return self._table.num_rows

        def __repr__(self):
            return '{0}({1})'.format(type(self).__name__, repr(self._table))


def _zero_on_type_error(column_fn):
    """Wrap a function on an np.ndarray to return 0 on a type error."""
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


class _Methods:
    def __init__(self, table, base):
        self.table = table
        self.base = base

    def __getattr__(self, attr):
        cs = self.table.columns
        if all(hasattr(c, self.base) and hasattr(getattr(c, self.base), attr) for c in cs):
            attrs = [getattr(getattr(c, self.base), attr) for c in cs]
            return _with_columns_wrapped(attrs, self.table)
        else:
            msg = "'{0}' object has no attribute '{1}'".format(type(self.table).__name__, attr)
            raise AttributeError(msg)


def _with_columns_wrapped(attrs, table):
    """Construct a table from the results of attrs, wrapping method calls."""
    if all(callable(attr) for attr in attrs):
        @functools.wraps(attrs[0])
        def method(*args, **vargs):
            """Create a table from the results of calling attrs."""
            columns = [attr(*args, **vargs) for attr in attrs]
            return table._with_columns(columns)
        return method
    else:
        return table._with_columns([[attr] for attr in attrs])
