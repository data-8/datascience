# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Provide a table of data regarding bachelor's degrees earned by women.

The data is broken down by field for any given year.

Licence: `CC0`_

Sourced from: https://www.kaggle.com/datasets/sureshsrinivas/bachelorsdegreewomenusa

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.degrees.data`

.. bokeh-sampledata-xref:: degrees

"""

from __future__ import annotations

from . import package_csv

__all__ = ("data",)

data = package_csv("percent-bachelors-degrees-women-usa.csv")
