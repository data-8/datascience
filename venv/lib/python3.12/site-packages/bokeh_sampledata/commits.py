# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Time series of commits for a GitHub user between 2012 and 2016.

License: `Public Domain`_

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.commits.data`

.. bokeh-sampledata-xref:: commits

"""

from __future__ import annotations

import pandas as pd

from . import package_csv

__all__ = ("data",)

data = package_csv(
    "commits.txt.gz",
    parse_dates=True,
    header=None,
    names=["day", "datetime"],
    index_col="datetime",
)
data.index = pd.to_datetime(data.index, utc=True).tz_convert("US/Central")
data["time"] = data.index.time
