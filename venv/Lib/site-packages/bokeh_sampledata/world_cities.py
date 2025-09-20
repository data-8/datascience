# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Names and locations of world cities with at least 5000 inhabitants.

License: `CC BY 2.0`_

Sourced from http://www.geonames.org/export/ (*cities5000.zip*)

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.world_cities.data`

.. bokeh-sampledata-xref:: world_cities

"""

from __future__ import annotations

from . import package_csv

__all__ = ("data",)

data = package_csv("world_cities.csv", na_filter=False)
