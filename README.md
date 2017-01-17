# datascience

A Berkeley library for introductory data science.

[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dsten/datascience?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Documentation Status](https://readthedocs.org/projects/datascience/badge/?version=master)](http://datascience.readthedocs.org/en/master/?badge=master)


*written by Professor [John DeNero](http://denero.org), Professor
[David Culler](http://www.cs.berkeley.edu/~culler),
[Sam Lau](https://github.com/samlau95), and [Alvin Wan](http://alvinwan.com)*

For an example of usage, see the [Berkeley Data 8 class](http://data8.org/).

[![Build Status](https://travis-ci.org/data-8/datascience.svg?branch=master)](https://travis-ci.org/data-8/datascience)
[![Coverage Status](https://coveralls.io/repos/dsten/datascience/badge.svg?branch=master&service=github)](https://coveralls.io/github/dsten/datascience?branch=master)

## Installation

Use `pip`:

```
pip install datascience
```

## Changelog

This project adheres to [Semantic Versioning](http://semver.org/).

### v0.10.0
- Added optional formatters to "Table.with_column", "Table.with_columns", and "Table.append_column".  

### v0.9.0
- Added "Table.remove"

### v0.8.2

- Added `proportions_from_distribution` method to `datascience.util`.
  (993e3d2)
- `Table.column` now throws a descriptive `ValueError` instead of a `KeyError`
  when the column isn't in the table. (ef8b319)

### v0.8.0
**Breaking changes**

- Change default behavior of `table.sample` to `with_replacement=True` instead
  of `False`. (3717b67)

**Additions**

- Added `Map.copy`.
- Added `Map.overlay` which overlays a feature(s) on a new copy of Map.
  (315bb63e)

### v0.7.1
- Remove rogue print from `table.hist`

### v0.7.0
- Added predicates for string comparison: `containing` and `contained_in`. (#231)

## Documentation

API reference is at http://data8.org/datascience/ .

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

After that, go ahead and start hacking!

The `source activate datascience` command must be run each time you develop in
the package. Alternatively, you can install [direnv][direnv] to auto-load/unload
the environment.

Documentation is generated from the docstrings in the methods and is pushed online
at http://data8.org/datascience/ automatically. If you want to preview the docs
locally, use these commands:

    make docs       # Generates docs inside doc/ folder
    make serve_docs # Starts a local server to view docs

## Using Zenhub

We use [Zenhub](https://www.zenhub.io/) to organize development on this library.
To get started, go ahead and install the [Zenhub Chrome Extension][zenhub-extension].

[zenhub-extension]: https://chrome.google.com/webstore/detail/zenhub-for-github/ogcgkffhplmphkaahpmffcafajaocjbd?hl=en-US

Then navigate to [the issue board](#boards) or press `b`. You'll see a screen
that looks something like this:

![screenshot 2015-09-24 23 03 57](https://cloud.githubusercontent.com/assets/2468904/10094128/ddc05b92-6310-11e5-9a23-d51216370e89.png)

- **New Issues** are issues that are just created and haven't been prioritized.
- **Backlogged** issues are issues that are not high priority, like nice-to-have
features.
- **To Do** issues are high priority and should get done ASAP, such as
breaking bugs or functionality that we need to lecture on soon.
- Once someone has been assigned to an issue, that issue should be moved into
the **In Progress** column.
- When the task is complete, we close the related issue.

### Example Workflow

1. John creates an issue called "Everything is breaking". It goes into the New
Issues pipeline at first.
2. This issue is important, so John immediately moves it into the To Do
pipeline. Since he has to go lecture for 61A, he doesn't assign it to himself
right away.
3. Sam sees the issue, assigns himself to it, and moves it into the In Progress
pipeline.
4. After everything is fixed, Sam closes the issue.

Here's another example.

1. Ani creates an issue asking for beautiful histograms. Like before, it goes
into the New Issues pipeline.
2. John decides that the issue is not as high priority right now because other
things are breaking, so he moves it into the Backlog pipeline.
3. When he has some more time, John assigns himself the issue and moves it into
the In Progress pipeline.
4. Once the issue is finished, he closes the issue.

## Publishing

```
python setup.py sdist upload -r pypi
```
