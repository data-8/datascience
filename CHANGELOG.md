## Changelog

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](http://semver.org/).

### v0.17.1

* Remove sphinx & nbsphinx from requirements.txt, as it is not needed during runtime
  - [Issue 523](https://github.com/data-8/datascience/issues/523)

### v0.17.0
* Includes additional plotly-based plotting methods `Table#iplot`, `Table#ibar`, `Table#ibarh`, `Table#igroup_bar`, `Table#igroup_barh`, `Table#iscatter`, `Table#ihist`, and `Table#iscatter3d`
* New static methods `Table#interactive_plots` and `Table#static_plots` that redirect `Table#plot` to `Table#iplot`, `Table#barh` to `Table#ibarh`, etc. with same arguments
* New method `Table#scatter3d` that is a wrapper for `Table#iscatter3d` but _does not_ implement a matplotlib plot and raises a `RuntimeError` if called when interactive plots are not enabled
* New plotly-charts.ipynb notebook demonstrating how to work with the interactive function added to the testing directory
* Enables markers to be shaded with hex colors using Folium's BeautifyIcon plugin
* Changed default marker's default icon from 'info-circle' to no icon
* Added additional keyword column names to `map_table` including 'color_scale', 'area_scale', and 'cluster_by'
* New options 'color_scale' and 'area_scale' draw color gradients for markers and area gradients for circles based on column values
* New 'cluster_by' option groups column by value and assigns a cluster marker to each group
* Includes geocoding function `get_coordinates` that assigns latitude and longitude coordinates for U.S. locations by city, state, county, and zip code
* Updated Maps.ipynb notebook to showcase new changes made to mapping
* Data 8-friendly `datascience` reference notebook added to documentation using nbsphinx

### v0.16.2
* Fix bug so Table.sort(descending=True) no longer reverses order of ties.
* Fix bug so Table.copy(shallow=False) performs a deepcopy.

### v0.16.1
* No longer support the `colors` argument for `Table#scatter`.  Use `group` instead.


### v0.15.10
* Include ipynb files in tests, and when measuring test coverage

### v0.15.9
* Changed the `radius` argument of a Circle (in Maps) to `area`.

### v0.15.8
* Fixes deprecation warnings with matplotlib's warn parameter.
* Small cleanups: update to latest version of folium.
* Add some additional documentation.

### v0.15.7
* Fixes bug with `Table#hist`.

### v0.15.6
* Adds support for NumPy v1.18.0+.

### v0.15.5
* Fixes multiple bugs with the `Table#remove`.

### v0.15.4
* Fixes bug with grouping tables with np.float64('nan') type objects.

### v0.15.3
* Fixed a bug that was introduced by v0.15.1.

### v0.15.2
* Adds more flexibility to `Marker#map_table` for user-defined options for markers.

### v0.15.1
* Fixed a bug related to histogram shading

### v0.15.0
* Added support for shading part of a histogram, e.g., for highlighting
the tail of a distribution.

### v0.14.1
* Adds optional argument to `Table#from_df` that keeps the index when
converting from a Pandas dataframe.

### v0.14.0
* Declares all dependencies required for this package in requirements.txt.

### v0.13.6
* Adds a warning to help students realize why `Table.with_columns(...)` doesn't work.

### v0.13.5
* Adds support for other built-in tile sets other than `OpenStreetMap` to `Map`.

### v0.13.4
* Adds support to `Map#read_geojson` for reading in GeoJSON from links.

### v0.13.3
* `make_array` automatically chooses int64 data type instead of int32 on Windows.

### v0.13.2
* Changes default formatting of numbers in printed output to 12345 instead of 12,345.

### v0.13.1
* Allows for the following notations ("floating arguments") to be used with
`Table#take` and `Table#exclude`: ex.`t.take(0, 1, 2, 3)` and `t.exclude(0, 2, 4)`.

### v0.13.0
* Removes deprecated argument for `Table#__init__`.

### v0.12.1
* Update mapping code to work with the latest version of Folium (0.9.1).

### v0.12.0
* Changes `Table#scatter`'s argument name of `colors` to `group` to mirror `Table#hist`.
* Makes a grouped scatterplot's legend identical to a group histogram's legend.

### v0.11.8
* Fixes bug where x-label doesn't show up for grouped histogram in certain conditions.

### v0.11.7
* Fixed bug where Table#hist was sometimes truncating the x-axis label.

### v0.11.6
* Fixes bug where error terms show up while plotting

### v0.11.5
* Fixes bug where joining tables that have columns that are already duplicated will sometimes join incorrectly.

### v0.11.4
* Fix bug where we warned inappropriately when passing a string to an `are.*` predicate.

### v0.11.3
* Switch from pandas.read_table to pandas.read_csv, to avoid deprecation warnings.  Shouldn't change the behavior of the library.

### v0.11.2
* `Table.append_column` now returns the table it is modifying.

### v0.11.1
* Add `shuffle` function to `Table`.

### v0.11.0
* Added `join` for multiple columns.

### v0.10.15
* Allow NumPy arrays to be appended into tables.

### v0.10.14
* Added optional formatters to "Table.with_column", "Table.with_columns", and "Table.append_column".

### v0.10.13
* Warning added for comparing iterables using predicates incorrectly.

### v0.10.12
* 'move_column' added.

### v0.10.11
* Created new methods 'first' and 'last'.

### v0.10.10
* 'append_column' now returns the table it is modifying.

### v0.10.9
* 'move_to_end' and 'move_to_start' can now take integer labels.

### v0.10.8
* Fixes test suite and removes all deprecated code in the test suite caused by deprecated API calls from the
datascience library.

### v0.10.7

* Adds `hist_of_counts` function

### v0.10.6

* Fixes minor issues introduced by matplotlib 2.x upgrade (https://github.com/data-8/datascience/pull/315)

### v0.10.5

* Fixes a bug in HTML table generation (https://github.com/data-8/datascience/pull/315)

### v0.10.4

* Add `sample_proportions` function.

### v0.10.3

* Fix `OrderedDict` bug in `Table.hist`.

### v0.10.2

* Fix `CurrencyFormatter` to handle commas.
* Fix `Table.hist` to keep histograms in the order of the columns.

### v0.10.1

* Fix `join` so that it keeps all rows in the inner join of two tables.

### v0.10.0

* Added `group_barh` and `group_bar` to plot counts by a grouping category,
  a common use case.
* Added options to `hist` to produce a histogram for each group on a
  column.
* Deprecated Table method `pivot_hist`. Added an option to `hist` to
  simulate `pivot_hist`'s behavior.

### v0.9.5

* DistributionFormatter added.

### v0.9.4

* Fix bug for relabeled columns that had a format already.

### v0.9.3

* Circles bound to values determine the circle area, not radius.

### v0.9.2

* Scatter diagrams can take data-driven size and color parameters.

### v0.9.1

* Changed signature of `apply`, `hist`, and `bin` to accept multiple columns without a list
* Deprecate `hist` argument name `counts` in favor of `bin_column`
* Rename various positional args (technically could break some code, but won't)
* Unified `with_column` and `with_columns` (not a breaking change)
* Unified `group` and `groups` (not a breaking change)

### v0.9.0

* Added "Table.remove"

### v0.8.2

* Added `proportions_from_distribution` method to `datascience.util`.
  (993e3d2)
* `Table.column` now throws a descriptive `ValueError` instead of a `KeyError`
  when the column isn't in the table. (ef8b319)

### v0.8.0

**Breaking changes**

* Change default behavior of `table.sample` to `with_replacement=True` instead
  of `False`. (3717b67)

**Additions**

* Added `Map.copy`.
* Added `Map.overlay` which overlays a feature(s) on a new copy of Map.
  (315bb63e)

### v0.7.1

* Remove rogue print from `table.hist`

### v0.7.0

* Added predicates for string comparison: `containing` and `contained_in`. (#231)
