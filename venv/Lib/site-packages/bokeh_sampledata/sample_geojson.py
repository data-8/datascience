# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Provide geojson data for the UK NHS England area teams.

License: `Open Government Licence`_

Sourced from https://github.com/JeniT/nhs-choices

A snapshot of data available from NHS Choices on November 14th, 2015.

.. bokeh-sampledata-xref:: sample_geojson

"""

from __future__ import annotations

from . import package_path

__all__ = ("geojson",)

with open(package_path("sample_geojson.geojson"), encoding="utf-8") as f:
    geojson = f.read()
