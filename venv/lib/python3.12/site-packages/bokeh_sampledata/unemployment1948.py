# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""US Unemployment rate data by month and year, from 1948 to 2013.

License: `Public Domain`_

Sourced from: https://www.bls.gov

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.unemployment1948.data`

.. bokeh-sampledata-xref:: unemployment1948

"""

from __future__ import annotations

from . import package_csv

__all__ = ("data",)

data = package_csv("unemployment1948.csv")
