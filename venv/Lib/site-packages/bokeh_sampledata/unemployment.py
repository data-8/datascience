# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Per-county unemployment data for Unites States in 2009.

License: `Public Domain`_

Sourced from: https://www.bls.gov

This module contains one dict: ``data``.

The dict is indexed by the two-tuples containing ``(state_id, county_id)`` and
has the unemployment rate (2009) as the value.

.. code-block:: ipython

    {
        (1, 1): 9.7,
        (1, 3): 9.1,
        ...
    }

.. bokeh-sampledata-xref:: unemployment

"""

from __future__ import annotations

import csv

from . import package_path

__all__ = ("data",)

data = {}

with open(package_path("unemployment09.csv"), encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=",", quotechar='"')
    for row in reader:
        _, state_id, county_id, _, _, _, _, _, rate = row
        data[(int(state_id), int(county_id))] = float(rate)
