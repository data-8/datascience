# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Time series of historical average sea surface temperatures.

License: free to use and redistribute (see `this table`_ for details).

Sourced from http://www.neracoos.org/erddap/tabledap/index.html (table *B01_sbe37_all*)

This module contains one pandas Dataframe: ``sea_surface_temperature``.

.. rubric:: ``sea_surface_temperature``

:bokeh-dataframe:`bokeh.sampledata.sea_surface_temperature.sea_surface_temperature`

.. bokeh-sampledata-xref:: sea_surface_temperature

.. _this table: http://www.neracoos.org/erddap/info/B01_sbe37_all/index.html

"""

from __future__ import annotations

from . import package_csv

__all__ = ("sea_surface_temperature",)

sea_surface_temperature = package_csv(
    "sea_surface_temperature.csv.gz",
    parse_dates=True,
    index_col=0,
)

sea_surface_temperature.rename(
    columns={"temperature (celsius)": "temperature"},
    inplace=True,
)

sea_surface_temperature.index.name = "time"
