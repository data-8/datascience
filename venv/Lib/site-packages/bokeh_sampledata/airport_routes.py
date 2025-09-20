# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Airport routes data from OpenFlights.org.

License: `ODbL 1.0`_

Sourced from `https:/openflights.org/data.html/` on September 07, 2017.
This historic link is not active anymore, but https:/openflights.org/ is
still available.

This module contains two pandas Dataframes: ``airports`` and ``routes``.

.. rubric:: ``airports``

:bokeh-dataframe:`bokeh.sampledata.airport_routes.airports`

.. rubric:: ``routes``

:bokeh-dataframe:`bokeh.sampledata.airport_routes.routes`

.. bokeh-sampledata-xref:: airport_routes

"""

from __future__ import annotations

from . import package_csv

__all__ = ("airports", "routes")

airports = package_csv("airports.csv")

routes = package_csv("routes.csv")
