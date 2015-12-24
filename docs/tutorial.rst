Start Here: ``datascience`` Tutorial
====================================

This is a brief introduction to the functionality in
:py:mod:`datascience`.  For a complete reference guide, please see
:ref:`tables-overview`.

For other useful tutorials and examples, see:

- `The textbook introduction to Tables`_
- `Example notebooks`_

.. _The textbook introduction to Tables: http://data8.org/text/1_data.html#tables
.. _Example notebooks: https://github.com/deculler/TableDemos

Getting Started
---------------

The most important functionality in the package is is the :py:class:`Table`
class, which is the structure used to represent columns of data. You may load
the class with:

.. ipython:: python

    from datascience import Table

In the IPython notebook, type ``Table.`` followed by the TAB-key to see a list
of members.

Note that for the Data Science 8 class we also import additional packages and
settings for all assignments and labs. This is so that plots and other available
packages mirror the ones in the textbook more closely. The exact code we use is:

.. code-block:: python

    # HIDDEN

    import matplotlib
    matplotlib.use('Agg')
    from datascience import Table
    %matplotlib inline
    import matplotlib.pyplot as plots
    import numpy as np
    plots.style.use('fivethirtyeight')

In particular, the lines involving ``matplotlib`` allow for plotting within the
IPython notebook.

Creating a Table
----------------

A Table is a sequence of labeled columns of data.

The basic Table constructor works as follows:

.. ipython:: python

    letters = ['a', 'b', 'c', 'z']
    counts = [9, 3, 3, 1]
    points = [1, 2, 2, 10]

    t = Table(columns=[letters, counts, points],
              labels=['letter', 'count', 'points'])

    print(t)

Note how the first keyword, ``columns``, specifies the contents of the table,
and how the second, ``labels``, gives a name to each column. See
:meth:`~datascience.tables.Table.__init__` for more details.

------

A table could also be read from a CSV file (that can be exported from an Excel
spreadsheet, for example).  Here's the content of an example file:

.. ipython:: python

    cat mydata.csv

And this is how we load it in as a :class:`Table` using
:meth:`~datascience.tables.Table.read_table`:

.. ipython:: python

    Table.read_table('sample.csv')

CSVs from URLs are also valid inputs to
:meth:`~datascience.tables.Table.read_table`:

.. ipython:: python

    Table.read_table('http://data8.org/text/sat2014.csv')

------

For convenience, you can also initialize a Table from a dictionary of column
names using
:meth:`~datascience.tables.Table.from_columns_dict`.

.. ipython:: python

    Table.from_columns_dict({
       'letter': letters,
       'count': counts,
       'points': points,
    })

This example illustrates the fact that built-in Python dictionaries don't
preserve their key order -- the dictionary keys are ordered ``'letter'``,
``'count'``, then ``'points'``, but the table columns are ordered ``'points'``,
``'count'``, then ``'letter'``). If you want to ensure the order of your
columns, use an ``OrderedDict``.

Accessing Values
----------------

To access values of columns in the table, use
:meth:`~datascience.tables.Table.values`.

.. ipython:: python

    t

    t.values('letter')
    t.values('count')

    t['letter'] # This is a shorthand for t.values('letter')

To access values by row, :meth:`~datascience.tables.Table.rows` returns an
list-like :class:`~datascience.tables.Table.Rows` object that contains
tuple-like :class:`~datascience.tables.Table.Row` objects.

.. ipython:: python

    t.rows
    t.rows[0]

    second = t.rows[1]
    second
    second[0]
    second[1]

To get the number of rows, use :attr:`~datascience.tables.Table.num_rows`.

.. ipython:: python

    t.num_rows


Manipulating Data
-----------------

Here are some of the most common operations on data. For the rest, see the
reference (:ref:`tables-overview`).

Adding a column with :meth:`~datascience.tables.Table.with_column`:

.. ipython:: python

    t
    t.with_column('vowel?', ['yes', 'no', 'no', 'no'])
    t # .with_column returns a new table without modifying the original

    t.with_column('2 * count', t['count'] * 2) # A simple way to operate on columns

Selecting columns with :meth:`~datascience.tables.Table.select`:

.. ipython:: python

    t.select('letter')
    t.select(['letter', 'points'])

Renaming columns with :meth:`~datascience.tables.Table.with_relabeling`:

.. ipython:: python

    t
    t.with_relabeling('points', 'other name')
    t
    t.with_relabeling(['letter', 'count', 'points'], ['x', 'y', 'z'])

Selecting out rows by index with :meth:`~datascience.tables.Table.take` and
conditionally with :meth:`~datascience.tables.Table.where`:

.. ipython:: python

    t
    t.take(2) # the third row
    t.take[0:2] # the first and second rows

.. ipython:: python

    t.where('points', 2) # rows where points == 2
    t.where(t['count'] < 8) # rows where count < 8

    t['count'] < 8 # .where actually takes in an array of booleans
    t.where([False, True, True, True]) # same as the last line

Operate on table data with :meth:`~datascience.tables.Table.sort`,
:meth:`~datascience.tables.Table.group`, and
:meth:`~datascience.tables.Table.pivot`

.. ipython:: python

    t
    t.sort('count')
    t.sort('letter', descending = True)

.. ipython:: python

    t.group('count')

    # You may pass a reducing function into the collect arg
    # Note the renaming of the points column because of the collect arg
    t.select(['count', 'points']).group('count', collect = sum)

.. ipython:: python

    other_table = Table([
        ['married', 'married', 'partner', 'partner', 'married'],
        ['Working as paid', 'Working as paid', 'Not working', 'Not working', 'Not working'],
        [1, 1, 1, 1, 1]
    ],
    ['mar_status', 'empl_status', 'count'])
    other_table

    other_table.pivot('mar_status', 'empl_status', 'count', collect = sum)

Visualizing Data
----------------

We'll start with some data drawn at random from two normal distributions:

.. ipython:: python

    normal_data = Table(
        [ np.random.normal(loc = 1, scale = 2, size = 100),
          np.random.normal(loc = 4, scale = 3, size = 100) ],
        ['data1', 'data2']
    })

    normal_data

Draw histograms with :meth:`~datascience.tables.Table.hist`:

.. ipython:: python

    @savefig hist.png width=4in
    normal_data.hist()

.. ipython:: python

    @savefig hist_binned.png width=4in
    normal_data.hist(bins = range(-5, 10))

.. ipython:: python

    @savefig hist_overlay.png width=4in
    normal_data.hist(bins = range(-5, 10), overlay = True)

If we treat the ``normal_data`` table as a set of x-y points, we can
:meth:`~datascience.tables.Table.plot` and
:meth:`~datascience.tables.Table.scatter`:

.. ipython:: python

    @savefig plot.png width=4in
    normal_data.sort('data1').plot('data1') # Sort first to make plot nicer

.. ipython:: python

    @savefig scatter.png width=4in
    normal_data.scatter('data1')

.. ipython:: python

    @savefig scatter_line.png width=4in
    normal_data.scatter('data1', fit_line = True)

Use :meth:`~datascience.tables.Table.barh` to display categorical data.

.. ipython:: python

    t
    t.barh('letter')

Exporting
---------

Exporting to CSV is the most common operation and can be done by first
converting to a pandas dataframe with :meth:`~datascience.tables.Table.to_df`:

.. ipython:: python

    normal_data

    # index = False prevents row numbers from appearing in the resulting CSV
    normal_data.to_df().to_csv('normal_data.csv', index = False)

An Example
----------
To come.

Drawing Maps
------------
To come.
