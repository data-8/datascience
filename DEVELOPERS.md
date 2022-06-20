## Developing

The required environment for installation and tests is the
[Anaconda Python3 distribution](http://continuum.io/downloads#py34)

If you encounter an `Image not found` error on **Mac OSX**, you may need an
[XQuartz upgrade](http://xquartz.macosforge.org/landing/).

Start by cloning this repository:

    git clone https://github.com/data-8/datascience

Install the dependencies into a [Conda environment][envs] with:

    conda env create -f osx_environment.yml -n datascience
    # For Linux, use
    conda env create -f linux_environment.yml -n datascience

[envs]: http://conda.pydata.org/docs/using/envs.html

Source the environment to use the correct packages while developing:

    source activate datascience
    # `source deactivate` will unload the environment

The above command must be run each time you develop in the package. You can also
install [direnv][direnv] to auto-load/unload the environment.

[direnv]: http://direnv.net/

Install `datascience` locally with:

    make install

Then, run the tests:

    make test

Alternatively you can use tox. Its used for testing the package against different python versions in one go. Just make sure you have the python versions available globally or in your virtual environment

Install tox

`pip install tox`

Run tests

`$ tox`

After that, go ahead and start hacking!

The `source activate datascience` command must be run each time you develop in
the package. Alternatively, you can install [direnv][direnv] to auto-load/unload
the environment.

Documentation is generated from the docstrings in the methods and is pushed online
at <http://data8.org/datascience/> automatically. If you want to preview the docs
locally, use these commands:

    make docs       # Generates docs inside doc/ folder
    make serve_docs # Starts a local server to view docs

## Publishing

```
python setup.py sdist
twine upload dist/*
```
