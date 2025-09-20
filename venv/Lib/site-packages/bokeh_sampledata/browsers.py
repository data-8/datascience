# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Browser market share by version from November 2013.

License: `CC BY-SA 3.0`_

Sourced from http://gs.statcounter.com/#browser_version-ww-monthly-201311-201311-bar

Icon images sourced from https://github.com/alrra/browser-logos

This module contains one pandas Dataframe: ``browsers_nov_2013``.

.. rubric:: ``browsers_nov_2013``

:bokeh-dataframe:`bokeh.sampledata.browsers.browsers_nov_2013`

The module also contains a dictionary ``icons`` with base64-encoded PNGs of the
logos for Chrome, Firefox, Safari, Opera, and IE.

.. bokeh-sampledata-xref:: browsers

"""

from __future__ import annotations

from os.path import join

import pandas as pd

from . import package_csv, package_path

__all__ = ("browsers_nov_2013", "icons")


def _read_data() -> pd.DataFrame:
    df = package_csv("browsers_nov_2013.csv", names=("Version", "Share"), skiprows=1)
    versions: pd.Series[str] = df["Version"].map(lambda x: x.rsplit(" ", 1))
    df["Browser"] = versions.map(lambda x: x[0])
    df["VersionNumber"] = versions.map(lambda x: x[1] if len(x) == 2 else "0")
    return df


def _read_icons() -> dict[str, bytes]:
    icons = {}
    for browser in ("Chrome", "Firefox", "Safari", "Opera", "IE"):
        icon_path = join("icons", f"{browser.lower()}_32x32.png")
        with open(package_path(icon_path), "rb") as icon:
            icons[browser] = icon.read()
    return icons


browsers_nov_2013 = _read_data()
icons = _read_icons()
