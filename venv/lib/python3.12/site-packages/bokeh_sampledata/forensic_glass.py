# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Correlations in mineral content for forensic glass samples.

License: `Public Domain`_

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.forensic_glass.data`

.. bokeh-sampledata-xref:: forensic_glass

"""

from __future__ import annotations

from . import package_csv

__all__ = ("data",)

data = package_csv("forensic_glass.csv")
