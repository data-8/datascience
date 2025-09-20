# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Exercise data (including altitude, speed and power) from a cycling workout.

License: `CC0`_

Sourced from personal data.

This module contains one pandas Dataframe: ``cycling``.

.. rubric:: ``cycling``

:bokeh-dataframe:`bokeh.sampledata.cycling.cycling`

.. bokeh-sampledata-xref:: cycling

"""

from __future__ import annotations

import pandas as pd

from . import package_csv

__all__ = ("cycling",)


def _read_data() -> pd.DataFrame:
    df = package_csv("cycling.csv", index_col=0)
    df["time"] = pd.to_timedelta(df["time"])
    return df


cycling = _read_data()
