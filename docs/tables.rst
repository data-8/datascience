.. _tables-overview:

Tables (``datascience.tables``)
===============================

.. currentmodule:: datascience.tables

**Summary of methods for Table. Click a method to see its documentation.**

One note about reading the method signatures for this page: each method is
listed with its arguments. However, optional arguments are specified in
brackets. That is, a method that's documented like

``Table.foo`` (first_arg, second_arg[, some_other_arg, fourth_arg])

means that the ``Table.foo`` method must be called with first_arg and second_arg
and optionally some_other_arg and fourth_arg. That means the following are valid
ways to call ``Table.foo``::

    some_table.foo(1, 2)
    some_table.foo(1, 2, 'hello')
    some_table.foo(1, 2, 'hello', 'world')
    some_table.foo(1, 2, some_other_arg='hello')

But these are not valid::

    some_table.foo(1) # Missing arg
    some_table.foo(1, 2[, 'hi']) # SyntaxError
    some_table.foo(1, 2[, 'hello', 'world']) # SyntaxError

If that syntax is confusing, you can click the method name itself to get to the
details page for that method. That page will have a more straightforward syntax.

At the time of this writing, most methods only have one or two sentences of
documentation, so what you see here is all that you'll get for the time being.
We are actively working on documentation, prioritizing the most complicated
methods (mostly visualizations).

Creation

.. autosummary::
    :toctree: _autosummary

    Table.__init__
    Table.empty
    Table.from_records
    Table.from_columns_dict
    Table.read_table
    Table.from_df
    Table.from_array

Extension (does not modify original table)

.. autosummary::
    :toctree: _autosummary

    Table.with_column
    Table.with_columns
    Table.with_row
    Table.with_rows
    Table.relabeled

Accessing values

.. autosummary::
    :toctree: _autosummary

    Table.num_columns
    Table.columns
    Table.column
    Table.num_rows
    Table.rows
    Table.row
    Table.labels
    Table.first
    Table.last
    Table.values
    Table.column_index
    Table.apply

Mutation (modifies table in place)

.. autosummary::
    :toctree: _autosummary

    Table.set_format
    Table.move_to_start
    Table.move_to_end
    Table.append
    Table.append_column
    Table.relabel
    Table.remove

Transformation (creates a new table)

.. autosummary::
    :toctree: _autosummary

    Table.copy
    Table.select
    Table.drop
    Table.take
    Table.exclude
    Table.move_column
    Table.where
    Table.sort
    Table.group
    Table.groups
    Table.pivot
    Table.stack
    Table.join
    Table.stats
    Table.percentile
    Table.sample
    Table.shuffle
    Table.sample_from_distribution
    Table.split
    Table.bin
    Table.pivot_bin
    Table.relabeled
    Table.with_row
    Table.with_rows
    Table.with_column
    Table.with_columns

Exporting / Displaying

.. autosummary::
    :toctree: _autosummary

    Table.show
    Table.as_text
    Table.as_html
    Table.index_by
    Table.to_array
    Table.to_df
    Table.to_csv

Visualizations

.. autosummary::
    :toctree: _autosummary

    Table.plot
    Table.bar
    Table.group_bar
    Table.barh
    Table.group_barh
    Table.pivot_hist
    Table.hist
    Table.hist_of_counts
    Table.scatter
    Table.scatter3d
    Table.boxplot
    Table.interactive_plots
    Table.static_plots
