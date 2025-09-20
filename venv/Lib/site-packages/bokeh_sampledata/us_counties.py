# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""This modules exposes geometry data for Unites States.

This module contains one dict: ``data``.

The data is indexed by two-tuples of ``(state_id, county_id)`` that
have the following dictionaries as values:

.. code-block:: ipython

    In [25]: data[(1,1)]
    Out[25]:
    {
        'name': 'Autauga',
        'detailed name': 'Autauga County, Alabama',
        'state': 'al',
        'lats': [32.4757, ..., 32.48112],
        'lons': [-86.41182, ..., -86.41187]
    }

Entries for ``'name'`` can have duplicates for certain states (e.g. Virginia).
The combination of ``'detailed name'`` and ``'state'`` will always be unique.

.. bokeh-sampledata-xref:: us_counties

"""

from __future__ import annotations

import csv
import xml.etree.ElementTree as et
from math import nan
from typing import TYPE_CHECKING, TypedDict

import numpy as np

from . import package_path

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ("data",)

# -----------------------------------------------------------------------------
# General API
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Dev API
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Private API
# -----------------------------------------------------------------------------


class CountyData(TypedDict):
    name: str
    detailed_name: str
    state: str
    lats: NDArray[np.float64]
    lons: NDArray[np.float64]


def _read_data() -> dict[tuple[int, int], CountyData]:
    data: dict[tuple[int, int], CountyData] = {}

    with open(package_path("US_Counties.csv"), encoding="utf-8") as f:
        next(f)
        reader = csv.reader(f, delimiter=",", quotechar='"')
        for row in reader:
            name, _, state, _, geometry, _, _, _, det_name, state_id, county_id, _, _ = row
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
            data[(int(state_id), int(county_id))] = CountyData(
                name=name,
                detailed_name=det_name,
                state=state,
                lats=np.array(lats),
                lons=np.array(lons),
            )

    return data


data = _read_data()
