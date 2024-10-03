# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Provide U.S. marriage and divorce statistics between 1867 and 2014

License: `Public Domain`_

Sourced from http://www.cdc.gov/nchs/

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.us_marriages_divorces.data`

.. bokeh-sampledata-xref:: us_marriages_divorces

"""

from __future__ import annotations

from . import package_csv

__all__ = ("data",)

data = package_csv("us_marriages_divorces.csv")
data = data.interpolate(method="linear", axis=0).ffill().bfill()
