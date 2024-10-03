# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Provide a periodic table data set.

License: `Public Domain`_

This module contains one pandas Dataframe: ``elements``.

.. rubric:: ``elements``

:bokeh-dataframe:`bokeh.sampledata.periodic_table.elements`

.. bokeh-sampledata-xref:: periodic_table

"""

from __future__ import annotations

from . import package_csv

__all__ = ("elements",)

elements = package_csv("elements.csv")
