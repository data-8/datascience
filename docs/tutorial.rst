Start Here: ``datascience`` Tutorial
====================================

This is a brief introduction to the functionality in
:py:mod:`datascience`.  For a complete reference guide, please see
:ref:`tables-overview`.

For other useful tutorials and examples, see:

- `The textbook introduction to Tables`_
- `Example notebooks`_

.. _The textbook introduction to Tables: https://www.inferentialthinking.com/chapters/06/Tables.html
.. _Example notebooks: https://github.com/deculler/TableDemos

.. contents:: Table of Contents
    :depth: 2
    :local:

Getting Started
---------------

The most important functionality in the package is is the :py:class:`Table`
class, which is the structure used to represent columns of data. First, load
the class:

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
    import matplotlib.pyplot as plt
    import numpy as np
    plt.style.use('fivethirtyeight')

In particular, the lines involving ``matplotlib`` allow for plotting within the
IPython notebook.

Creating a Table
----------------

A Table is a sequence of labeled columns of data.

A Table can be constructed from scratch by extending an empty table with
columns.

.. ipython:: python

    t = Table().with_columns(
        'letter', ['a', 'b', 'c', 'z'],
        'count',  [  9,   3,   3,   1],
        'points', [  1,   2,   2,  10],
    )

    print(t)

------

More often, a table is read from a CSV file (or an Excel spreadsheet).  Here's
the content of an example file:

.. ipython:: python

    cat sample.csv

And this is how we load it in as a :class:`Table` using
:meth:`~datascience.tables.Table.read_table`:

.. ipython:: python

    Table.read_table('sample.csv')

CSVs from URLs are also valid inputs to
:meth:`~datascience.tables.Table.read_table`:

.. ipython:: python

    Table.read_table('https://www.inferentialthinking.com/data/sat2014.csv')

------

It's also possible to add columns from a dictionary. For users< Python 3.6 this method is discouraged because dictionary does not preserve the order.

**Note**- New Dict implementation in Python 3.6 makes it possible for maintaining order of columns in dictionaires. 
          For more info- Read https://docs.python.org/3.6/whatsnew/3.6.html#whatsnew36-pep468


.. ipython:: python

    t = Table().with_columns({
        'letter': ['a', 'b', 'c', 'z'],
        'count':  [  9,   3,   3,   1],
        'points': [  1,   2,   2,  10],
    })

    print(t)

Accessing Values
----------------

To access values of columns in the table, use
:meth:`~datascience.tables.Table.column`, which takes a column label or index
and returns an array. Alternatively, :meth:`~datascience.tables.Table.columns`
returns a list of columns (arrays).

.. ipython:: python

    t

    t.column('letter')
    t.column(1)

You can use bracket notation as a shorthand for this method:

.. ipython:: python

    t['letter'] # This is a shorthand for t.column('letter')
    t[1]        # This is a shorthand for t.column(1)

To access values by row, :meth:`~datascience.tables.Table.row` returns a
row by index. Alternatively, :meth:`~datascience.tables.Table.rows` returns an
list-like :class:`~datascience.tables.Table.Rows` object that contains
tuple-like :class:`~datascience.tables.Table.Row` objects.

.. ipython:: python

    t.rows
    t.rows[0]
    t.row(0)

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

Renaming columns with :meth:`~datascience.tables.Table.relabeled`:

.. ipython:: python

    t
    t.relabeled('points', 'other name')
    t
    t.relabeled(['letter', 'count', 'points'], ['x', 'y', 'z'])

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

    # You may pass a reducing function into the collect arg
    # Note the renaming of the points column because of the collect arg
    t.select(['count', 'points']).group('count', collect=sum)

.. ipython:: python
    :okwarning:

    other_table = Table().with_columns(
        'mar_status',  ['married', 'married', 'partner', 'partner', 'married'],
        'empl_status', ['Working as paid', 'Working as paid', 'Not working',
                        'Not working', 'Not working'],
        'count',       [1, 1, 1, 1, 1])
    other_table

    other_table.pivot('mar_status', 'empl_status', 'count', collect=sum)

Visualizing Data
----------------

We'll start with some data drawn at random from two normal distributions:

.. ipython:: python

    normal_data = Table().with_columns(
        'data1', np.random.normal(loc = 1, scale = 2, size = 100),
        'data2', np.random.normal(loc = 4, scale = 3, size = 100))

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
    @savefig barh.png width=4in
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

We'll recreate the steps in `Chapter 12 of the textbook`_ to see if there is a
significant difference in birth weights between smokers and non-smokers using a
bootstrap test.

For more examples, check out `the TableDemos repo`_.

.. _Chapter 12 of the textbook: https://www.inferentialthinking.com/chapters/12/1/AB_Testing.html
.. _the TableDemos repo: https://github.com/deculler/TableDemos

From the text:

    The table ``baby`` contains data on a random sample of 1,174 mothers and
    their newborn babies. The column ``Birth Weight`` contains the birth weight
    of the baby, in ounces; ``Gestational Days`` is the number of gestational
    days, that is, the number of days the baby was in the womb. There is also
    data on maternal age, maternal height, maternal pregnancy weight, and
    whether or not the mother was a smoker.

.. ipython:: python

    baby = Table.read_table('https://www.inferentialthinking.com/data/baby.csv')
    baby # Let's take a peek at the table

    # Select out columns we want.
    smoker_and_wt = baby.select(['Maternal Smoker', 'Birth Weight'])
    smoker_and_wt

Let's compare the number of smokers to non-smokers.

.. ipython:: python

    smoker_and_wt.select('Maternal Smoker').group('Maternal Smoker')

We can also compare the distribution of birthweights between smokers and
non-smokers.

.. ipython:: python

    # Non smokers
    # We do this by grabbing the rows that correspond to mothers that don't
    # smoke, then plotting a histogram of just the birthweights.
    @savefig not_m_smoker_weights.png width=4in
    smoker_and_wt.where('Maternal Smoker', 0).select('Birth Weight').hist()

    # Smokers
    @savefig m_smoker_weights.png width=4in
    smoker_and_wt.where('Maternal Smoker', 1).select('Birth Weight').hist()

What's the difference in mean birth weight of the two categories?

.. ipython:: python

    nonsmoking_mean = smoker_and_wt.where('Maternal Smoker', 0).column('Birth Weight').mean()
    smoking_mean = smoker_and_wt.where('Maternal Smoker', 1).column('Birth Weight').mean()

    observed_diff = nonsmoking_mean - smoking_mean
    observed_diff

Let's do the bootstrap test on the two categories.

.. ipython:: python

    num_nonsmokers = smoker_and_wt.where('Maternal Smoker', 0).num_rows
    def bootstrap_once():
        """
        Computes one bootstrapped difference in means.
        The table.sample method lets us take random samples.
        We then split according to the number of nonsmokers in the original sample.
        """
        resample = smoker_and_wt.sample(with_replacement = True)
        bootstrap_diff = resample.column('Birth Weight')[:num_nonsmokers].mean() - \
            resample.column('Birth Weight')[num_nonsmokers:].mean()
        return bootstrap_diff

    repetitions = 1000
    bootstrapped_diff_means = np.array(
        [ bootstrap_once() for _ in range(repetitions) ])

    bootstrapped_diff_means[:10]

    num_diffs_greater = (abs(bootstrapped_diff_means) > abs(observed_diff)).sum()
    p_value = num_diffs_greater / len(bootstrapped_diff_means)
    p_value


Drawing Maps
------------
The main class in the maps module is the Map class. In this code we create a default Map. Maps can be displayed or converted to html.

.. ipython:: python

    from datascience.maps import Map    # import the Map class
    default_map = Map()                 # generate a default Map
    default_map.show()                  # display the Map
    
    html = default_map.as_html()        # generate the html
    with open('map.html', 'w') as f:    # make a file to store the html
        f.write(html)                   # write the html to the file

The maps modules also allows you to make custom maps with markers, circles and regions.

.. ipython:: python
    from datascience.maps import Map, Marker, Circle, Region            # import the Map, Marker, Circle and Region class
    
    # generates markers with custom sets of coordinates, colors and popups
    marker1 = Marker(37.372, -121.758, color="green", popup="My green marker") 
    marker2 = Marker(37.572, -121.758, color="orange", popup="My orange marker") 
    
    # generates a circle with a custom set of coordinates, color and popup
    circle = Circle(37.5, -122, color="red", area=1000, popup="My Circle")

    # make a geojson object which is needed when making a region
    geojson = {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [                                                # specifies the coordinates
            [[-121,37],[-121.5,37],[-121.5,37.5],[-121,37.5],[-121,37]] # these coordinates make a rectangle
        ]
      }
    }
    
    # make a region with your geojson object
    region = Region(geojson)
    
    
    # Initialize the map  
    custom_map = Map(features=[marker1, marker2, circle, region],       # specifies the features
             width=800,                                                 # specifies a custom width
             height=600                                                 # specifies a custom height
                    )
    custom_map.show()                                                   # display the map
