# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Route data (including altitude) for a bike race in Eastern Europe.

Sourced from https://bikemaraton.com.pl

This module contains one pandas Dataframe: ``obiszow_mtb_xcm``.

.. rubric:: ``obiszow_mtb_xcm``

:bokeh-dataframe:`bokeh.sampledata.mtb.obiszow_mtb_xcm`

.. bokeh-sampledata-xref:: mtb

"""

from __future__ import annotations

from . import package_csv

__all__ = ("obiszow_mtb_xcm",)

obiszow_mtb_xcm = package_csv("obiszow_mtb_xcm.csv")
