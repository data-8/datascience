# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Provide the Sample Superstore data set.

License: `CC0`_

Sourced from: https://www.kaggle.com/datasets/arshid/iris-flower-dataset

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.sample_superstore.data`

.. bokeh-sampledata-xref:: sample_superstore

"""

from __future__ import annotations

from . import package_csv

__all__ = ("data",)

data = package_csv("SampleSuperstore.csv")
