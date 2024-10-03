# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Butterfat percentage in the milk of five cattle breeds.

License: `Public Domain`_

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.cows.data`

.. bokeh-sampledata-xref:: cows

"""

from __future__ import annotations

from . import package_csv

__all__ = ("data",)

data = package_csv("cows.csv")
