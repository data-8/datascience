# datascience

A Berkeley library for introductory data science.

_written by Professor [John DeNero](http://denero.org), Professor
[David Culler](http://www.cs.berkeley.edu/~culler),
[Sam Lau](https://github.com/samlau95), and [Alvin Wan](http://alvinwan.com)_

For an example of usage, see the [Berkeley Data 8 class](http://data8.org/).

[![Documentation Status](https://readthedocs.org/projects/datascience/badge/?version=master)](http://datascience.readthedocs.org/en/master/?badge=master)
[![Build Status](https://github.com/data-8/datascience/actions/workflows/run_tests.yml/badge.svg?branch=master)](https://github.com/data-8/datascience/actions/workflows/run_tests.yml)
[![Coverage Status](https://coveralls.io/repos/data-8/datascience/badge.svg?branch=master&service=github)](https://coveralls.io/github/data-8/datascience?branch=master)

## Installation

Use `pip` to install the package: 
```
pip install datascience

```

To verify that the package is installed correctly, run:
```
python -c "import datascience; print(datascience.__version__)"

```

## Quick Start Guide 

After installing the package, you can start using datascience by importing it in Python:

```
from datascience import Table

# Create a simple table
data = Table().with_columns(
    "Name", ["Alice", "Bob", "Charlie"],
    "Age", [25, 30, 35]
)

# Display the table
data.show()
```

## Basic Data Manipulation

# Adding a new column
```
data = data.with_column("Height (cm)", [165, 180, 175])
```
# Sorting the table by age
```
sorted_data = data.sort("Age", descending=True)
sorted_data.show()
```

## Key Functions and Methods

# Table Creation
- Table() : Creates an empty table 
- Table.with_columns(column_name, values, ...) : Adds multiple columns to a table

# Data Manipulation
- Table.with_column(column_name, values) : Adds a single column
- Table.drop(column_name) :  Removes a column from the table.
- Table.sort(column_name, descending=False) : Sorts rows based on a column.

# Data Visualization 
- Table.plot(column_x, column_y) : Plots a graph using two columns.
- Table.hist(column) : Generates a histogram.
- Table.scatter(column_x, column_y) :  Creates a scatter plot. 

## Troubleshooting Guide

# 1. Installation Issues
Problem: ModuleNotFoundError: No module named 'datascience'
Solution: Ensure the package is installed using:

```
pip install --upgrade datascience
```

# 2. Import Errors
Problem: ImportError: cannot import name 'Table' from 'datascience'
Solution: Try the following:

Verify installation by running:
```
python -c "import datascience; print(datascience.__version__)"
```

# 3. Display Issues in Jupyter Notebook
Problem: Tables are not displaying correctly in Jupyter Notebook.
Solution:

```
pip install ipython notebook
```




