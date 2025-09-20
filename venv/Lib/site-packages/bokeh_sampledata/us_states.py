# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Geometry data for US States.

This module contains one dict: ``data``.

The data is indexed by the two letter state code (e.g., 'CA', 'TX') and has the
following structure:

.. code-block:: python

    In [4]: data["OR"]
    Out[4]:
    {
        'name': 'Oregon',
        'region': 'Northwest',
        'lats': [46.29443, ..., 46.26068],
        'lons': [-124.03622, ..., -124.15935]
    }

.. bokeh-sampledata-xref:: us_states

"""

from __future__ import annotations

import codecs
import csv
import gzip
import xml.etree.ElementTree as et
from math import nan
from typing import TYPE_CHECKING, TypedDict

import numpy as np

from . import package_path

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ("data",)


class StateData(TypedDict):
    name: str
    region: str
    lats: NDArray[np.float64]
    lons: NDArray[np.float64]


def _read_data() -> dict[str, StateData]:
    data = {}

    with gzip.open(package_path("US_Regions_State_Boundaries.csv.gz")) as f:
        decoded = codecs.iterdecode(f, "utf-8")
        next(decoded)
        reader = csv.reader(decoded, delimiter=",", quotechar='"')
        for row in reader:
            region, name, code, geometry, _ = row
            xml = et.fromstring(geometry)
            lats: list[float] = []
            lons: list[float] = []
            for i, poly in enumerate(xml.findall(".//outerBoundaryIs/LinearRing/coordinates")):
                if i > 0:
                    lats.append(nan)
                    lons.append(nan)
                assert isinstance(poly.text, str)
                coords = (c.split(",")[:2] for c in poly.text.split())
                lat, lon = list(
                    zip(*[(float(lat), float(lon)) for lon, lat in coords], strict=False),
                )
                lats.extend(lat)
                lons.extend(lon)
            data[code] = StateData(
                name=name,
                region=region,
                lats=np.array(lats),
                lons=np.array(lons),
            )

    return data


data = _read_data()
