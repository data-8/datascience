# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""A CSV timeseries of blood glucose measurements.

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.glucose.data`

.. bokeh-sampledata-xref:: glucose

"""

from __future__ import annotations

from . import package_csv

__all__ = ("data",)

data = package_csv("CGM.csv", sep=",", parse_dates=[1], index_col=1)
