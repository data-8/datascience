# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Mean daily temperatures in Lincoln, Nebraska, 2016.

License: `Public Domain`_

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.lincoln.data`

.. bokeh-sampledata-xref:: lincoln

"""

from __future__ import annotations

from . import package_csv

__all__ = ("data",)

data = package_csv("lincoln_weather.csv")
