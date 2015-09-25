Tables (``datascience.tables``)
========

.. currentmodule:: datascience.tables

**Summary of methods for Table. Click a method to see its documentation.**

At the time of this writing, most methods only have one or two sentences of
documentation, so what you see here is all that you'll get for the time being.
We are actively working on documentation, prioritizing the most
complicated methods (mostly visualizations).

Creation

.. autosummary::
    :toctree: _autosummary

    Table.__init__
    Table.from_rows
    Table.from_records
    Table.read_table


Accessing values

.. autosummary::
    :toctree: _autosummary

    Table.num_rows
    Table.rows
    Table.column_labels
    Table.columns

    Table.column_index
    Table.apply


Mutation (modifies table in place)

.. autosummary::
    :toctree: _autosummary

    Table.set_format
    Table.move_to_start
    Table.move_to_end
    Table.append
    Table.relabel

Transformation (creates a new table)

.. autosummary::
    :toctree: _autosummary

    Table.copy
    Table.select
    Table.drop
    Table.take
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
    Table.split

Exporting / Displaying

.. autosummary::
    :toctree: _autosummary

    Table.show
    Table.as_text
    Table.as_html
    Table.matrix
    Table.index_by

Visualizations

.. autosummary::
    :toctree: _autosummary

    Table.plot
    Table.barh
    Table.pivot_hist
    Table.hist
    Table.points
