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

## Setting up Local Development Environment

### For Windows Users

#### Setting up Windows Subsystem for Linux (WSL)

Before proceeding, it's recommended to set up Windows Subsystem for Linux (WSL) for better compatibility with development tools. Follow the official Microsoft documentation to install and set up WSL: [Install Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/install)

Once WSL is set up, open your preferred Linux distribution and continue with the following steps.

#### Installing Required Dependencies

Ensure you have Python, pip, and other necessary tools installed within your WSL environment.

sudo apt update
sudo apt install python3 python3-pip

### Generating local test report

1. Clone the repository:
git clone https://github.com/<your_username>/datascience.git

2. Navigate to the project directory
cd datascience

3. Install required packages:
pip install -r requirements.txt

4. Run tests using pytest
python3 pytest

5. Generate coverage report:
python3 coverage run -m pytest

6. View coverage report:
python3 coverage report

Optionally, integrate with Coveralls for code coverage tracking:
python3 coveralls

That's it! You have now successfully set up your local development environment and generated a test report.

After reviewing the results, you can determine which of the files under the source code folder need more tests based on their corresponding percentages.

Our aim is to have 97% of overall codebase coverage to ensure that there are no discrepencies.

Thank you for considering to help our project, we really appreciate it :)
