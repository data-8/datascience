# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""CO2 emmisions of selected countries in the years from 1950 to 2012.
Note that not all countries have values for the whole time range.

License: `Public Domain`_

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.emissions.data`

.. bokeh-sampledata-xref:: emissions

"""

from __future__ import annotations

from . import package_csv

__all__ = ("data",)

data = package_csv("emissions.csv")
