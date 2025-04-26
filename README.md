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

Use `pip`:

```
pip install datascience
```

A log of all changes can be found in CHANGELOG.md.

## Usage

To use the `datascience` library, you can import it in your Python scripts or Jupyter Notebooks.

```python
from datascience import Table

# Create a table with data
data = Table().with_columns(
  'A', [1, 2, 3, 4],
  'B', [5, 6, 7, 8]
)

# Display the table
data.show()

