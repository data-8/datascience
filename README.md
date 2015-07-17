# datascience

A library for introductory data science.

*written by Professor [John DeNero](http://denero.org), Professor [David Culler](http://www.cs.berkeley.edu/~culler), and [Alvin Wan](http://alvinwan.com)*

## Installation

Use `pip`:

```
pip install datascience
```

If you clone this repository, you may run tests against `datascience`:

```
python setup.py test
```

The recommended environment for installation and tests is the
[Anaconda Python3 distribution](http://continuum.io/downloads#py34)

If you encounter an `Image not found` error on **Mac OSX**, you may need an [XQuartz upgrade](http://xquartz.macosforge.org/landing/).

## Publishing

```
python setup.py sdist upload -r pypi
```
