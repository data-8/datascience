# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Demographic details of the passengers on board of the Titanic.

License: `Public Domain`_

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.titanic.data`

.. bokeh-sampledata-xref:: titanic

"""

from __future__ import annotations

from . import package_csv

__all__ = ("data",)

data = package_csv("titanic_all.csv")
