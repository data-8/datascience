# Developing

Developing this package requires the use of [Conda environments][envs] to manage the required additional packages used for testing and installion.

[envs]: http://conda.pydata.org/docs/using/envs.html

If `conda` is not installed on your local computer, you can easily install the [Anaconda Python3 distribution](http://continuum.io/downloads).

**Note**: If you encounter an `Image not found` error on **Mac OSX**, you may need an [XQuartz upgrade](http://xquartz.macosforge.org/landing/).

## Clone this repository
Start by cloning this repository:

    git clone https://github.com/data-8/datascience


## Install dependencies
Install the dependencies into a new `conda` environment named `datascience` with:

### For MacOS X
    conda env create -f osx_environment.yml -n datascience
    
### For Linux
    conda env create -f linux_environment.yml -n datascience

## Activate environment
Source the environment to use the correct packages while developing:

    source activate datascience
    
**Note**: The above command *must* be run each time you begin development on the package to activate the environment. You can install [direnv][direnv] to auto-load/unload the environment, but this is not required.

You can unload the environment when done developing with:

    source deactivate

## Testing
Currently, this project uses `pytest` for testing. To run the included tests:

First, install additional requirements for testing into the active environment with:

    pip install -r requirements-tests.txt
    
Then, install `datascience` locally with:

    make install

Then, run the tests:

    make test

### Alternative testing method (experimental)
You can also use `tox` for testing. `tox` can be used for testing the package against several python versions all at once. ou must ensure that the python versions you're testing against are available globally or in your virtual environment. Currently, [`tox.ini`](https://github.com/taylorgibson/datascience/blob/master/tox.ini#L7) is configured to test against 3.6, 3.7, and 3.8.

To test using `tox`:

First, install `tox`

    pip install tox

Then, run tests with:

    tox

**Note**: Testing with `tox` is currently experimental, and `tox` is not used in the Github action that runs on new commits and pull requests. Use at your own risk!

## Documentation
Documentation is generated from the docstrings in the methods and is pushed online
at <http://data8.org/datascience/> automatically. If you want to preview the docs
locally, use these commands:

    make docs       # Generates docs inside doc/ folder
    make serve_docs # Starts a local server to view docs

## Publishing

```
python setup.py sdist
python setup.py bdist_wheel
twine upload dist/*
```
