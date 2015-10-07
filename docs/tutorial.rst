Start Here: ``datascience`` Tutorial
====================================

This is general introduction to the functionality in
:py:mod:`datascience`.  For a reference guide, please see
:ref:`tables-overview`.

Introduction
------------

First, load the :py:mod:`datascience` module:

.. ipython:: python

   import datascience as ds

In the IPython notebook, type `ds.` followed by the TAB-key to see a list of members.
The most important of these is the :py:class:`Table` class, which is
the structure used to represent columns of data.

Basic Table Usage
-----------------

A table is constructed as follows:

.. ipython:: python

   letters = ['a', 'b', 'c', 'z']
   counts = [9, 3, 3, 1]
   points = [1, 2, 2, 10]

   t = ds.Table(columns=[letters, counts, points],
                labels=['letter', 'count', 'points'])

   print(t)

Note how the first keyword, ``columns``, specifies the contents of the
table, and how the second, ``labels``, gives a name to each column.

A table could also be read from a CSV file (that can be exported from
an Excel spreadsheet, for example).  Here's the content of the file:

.. ipython:: python

   cat mydata.csv

And this is how we load it in as a :class:`Table`:

.. ipython:: python

   t = ds.Table.read_table('mydata.csv')
   print(t)

More Advanced Table Usage
-------------------------

Once a table has been constructed, we can do various queries on it.

Print the first two entries:

.. ipython:: python

   print(t[:2])

Drawing Maps
------------
