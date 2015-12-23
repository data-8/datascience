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

    t = Table.read_table('sample.csv')
    print(t)

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
preserve their key order -- the dictionary keys are ordered 'letter', 'count',
then 'points', but the table columns are ordered 'points', 'count', then
'letter'). If you want to ensure the order of your columns, use an
``OrderedDict``.

Accessing Values
----------------

To access values of columns in the table, use
:meth:`~datascience.tables.Table.values`.

.. ipython:: python

    t

    t.values('x')
    t.values('y')

    t['x'] # This is a shorthand for t.values('x')

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

To get the number of rows, use :meth:`~datascience.tables.Table.num_rows`.

.. ipython:: python

    t.num_rows


Manipulating Data
-----------------



Visualizing Data
----------------
To come.

An Example
----------
To come.

Drawing Maps
------------
To come.
