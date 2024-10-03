# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""The four data series that comprise `Anscombe's Quartet`_.

License: `CC BY-SA 3.0`_

Sourced from: https://en.wikipedia.org/wiki/Anscombe%27s_quartet

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.anscombe.data`

.. bokeh-sampledata-xref:: anscombe

.. _Anscombe's Quartet: https://en.wikipedia.org/wiki/Anscombe%27s_quartet

"""

from __future__ import annotations

from . import package_csv

__all__ = ("data",)

data = package_csv("anscombe.csv", skipinitialspace=True)
