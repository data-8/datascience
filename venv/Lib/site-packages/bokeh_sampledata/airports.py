# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""US airports with field elevations > 1500 meters.

License: `Public Domain`_

Sourced from USGS service `http://services.nationalmap.gov` on October 15, 2015.
In the meantime this service was deactivated and is not available anymore.

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.airports.data`

.. bokeh-sampledata-xref:: airports

"""

from __future__ import annotations

import json

import pandas as pd

from . import package_path

__all__ = ("data",)

with open(package_path("airports.json"), "rb") as f:
    airports = json.load(f)
    data = pd.json_normalize(
        airports["features"],
        meta=[
            ["attributes", "nam"],
            ["attributes", "zv3"],
            ["geometry", "x"],
            ["geometry", "y"],
        ],
    )

    data = data.rename(
        columns={
            "attributes.nam": "name",
            "attributes.zv3": "elevation",
            "geometry.x": "x",
            "geometry.y": "y",
        },
    )
