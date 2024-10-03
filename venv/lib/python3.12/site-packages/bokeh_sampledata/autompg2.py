# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""A version of the Auto MPG data set.

License: `CC0`_

Sourced from https://archive.ics.uci.edu/ml/datasets/auto+mpg

This module contains one pandas Dataframe: ``autompg``.

.. rubric:: ``autompg2``

:bokeh-dataframe:`bokeh.sampledata.autompg2.autompg2`

.. bokeh-sampledata-xref:: autompg2

"""

from __future__ import annotations

from . import package_csv

__all__ = ("autompg2",)


def _capitalize_words(string: str) -> str:
    return " ".join(word.capitalize() for word in string.split(" "))


autompg2 = package_csv("auto-mpg2.csv")
autompg2["manufacturer"] = autompg2["manufacturer"].map(_capitalize_words)
autompg2["model"] = autompg2["model"].map(_capitalize_words)
autompg2["drv"] = autompg2["drv"].replace({"f": "front", "r": "rear", "4": "4x4"})
