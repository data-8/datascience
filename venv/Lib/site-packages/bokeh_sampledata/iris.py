# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Provide `Fisher's Iris dataset`_.

License: `CC0`_

Sourced from: https://www.kaggle.com/datasets/arshid/iris-flower-dataset

This module contains one pandas Dataframe: ``flowers``.

.. note::
    This sampledata is maintained for historical compatibility. Please consider
    `alternatives to Iris`_ such as :ref:`sampledata_penguins`.

.. rubric:: ``flowers``

:bokeh-dataframe:`bokeh.sampledata.iris.flowers`

.. bokeh-sampledata-xref:: iris

.. _Fisher's Iris dataset: https://en.wikipedia.org/wiki/Iris_flower_data_set
.. _alternatives to Iris: https://www.meganstodel.com/posts/no-to-iris/

"""

from __future__ import annotations

from . import package_csv

__all__ = ("flowers",)

flowers = package_csv("iris.csv")
