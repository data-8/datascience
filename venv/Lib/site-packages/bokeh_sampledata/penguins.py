# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Provide data from the `Palmer Archipelago (Antarctica) penguin dataset`_.

License: `CC0`_

Sourced from https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.penguins.data`

.. bokeh-sampledata-xref:: penguins

.. _Palmer Archipelago (Antarctica) penguin dataset: https://github.com/allisonhorst/palmerpenguins

"""

from __future__ import annotations

from . import package_csv

__all__ = ("data",)

data = package_csv("penguins.csv")
