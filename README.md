

[![Documentation Status](https://readthedocs.org/projects/datascience/badge/?version=master)](http://datascience.readthedocs.org/en/master/?badge=master)
[![Build Status](https://github.com/data-8/datascience/actions/workflows/run_tests.yml/badge.svg?branch=master)](https://github.com/data-8/datascience/actions/workflows/run_tests.yml)
[![Coverage Status](https://coveralls.io/repos/data-8/datascience/badge.svg?branch=master&service=github)](https://coveralls.io/github/data-8/datascience?branch=master)

# DataScience

DataScience is a Python library designed to explore different classes for managing and analyzing data.

 - Written by Professor John DeNero, Professor David Culler, Sam Lau, and Alvin Wan

Table of Contents

1. [Installation](#installation)
   - [Using pip](#using-pip)
   - [GitHub](#github)
2. [Dependencies](#dependencies)
3. [Developing](#developing)
 
   - [Activating Environment]
   - [Deactivate Environment]
4. [Usage](#usage)




## Installation


### GitHub
Clone the following repository
```
git clone https://github.com/data-8/datascience.git
```
### Using pip

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the DataScience library.

```bash
pip install datascience
```


## Dependencies

This project requires Python 3.6+ and the following key libraries:

- `matplotlib`
- `numpy`
- `pandas`
- `scipy`
- `pytest` (for testing)

### Full Dependency List

For the full list of dependencies, see the [`requirements.txt`](https://github.com/data-8/datascience/blob/main/requirements.txt) and [`environment.yml`](https://github.com/data-8/datascience/blob/main/environment.yml)
## Developing 
Install the dependencies into a new conda environment named datascience.

Activating Environment
```
source activate datascience
```

Deactivate Environment
```
source deactivate
```
## Usage 
```
from datascience import (here)
```
Select the class that you want to use.
```
from datascience import *
```
You can select all the classes with " * "

Example
```
Table().with_columns(
    'cars', make_array(1, 2, 3),
    'colors', make_array('red', 'blue', 'black')
)
```
