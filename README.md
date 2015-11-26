# datascience

[![Join the chat at https://gitter.im/dsten/datascience](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dsten/datascience?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

A library for introductory data science.

*written by Professor [John DeNero](http://denero.org), Professor
[David Culler](http://www.cs.berkeley.edu/~culler),
[Sam Lau](https://github.com/samlau95), and [Alvin Wan](http://alvinwan.com)*

[![Build Status](https://travis-ci.org/dsten/datascience.svg?branch=master)](https://travis-ci.org/dsten/datascience)
[![Coverage Status](https://coveralls.io/repos/dsten/datascience/badge.svg?branch=master&service=github)](https://coveralls.io/github/dsten/datascience?branch=master)

## Installation

Use `pip`:

```
pip install datascience
```

## Developing

The recommended environment for installation and tests is the
[Anaconda Python3 distribution](http://continuum.io/downloads#py34)

If you encounter an `Image not found` error on **Mac OSX**, you may need an
[XQuartz upgrade](http://xquartz.macosforge.org/landing/).

Start by cloning this repository:

    git clone https://github.com/dsten/datascience

Install it locally with:

    make install

Then, run the tests:

    make test

After that, go ahead and start hacking!

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
