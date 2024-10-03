# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Provide medal counts by country for the 2014 Olympics.

Sourced from public news sources in 2014.

This module contains a single dict: ``data``.

The dictionary has a key ``"data"`` that lists sub-dictionaries, one for each
country:

.. code-block:: python

    {
        'abbr': 'DEU',
        'medals': {'total': 15, 'bronze': 4, 'gold': 8, 'silver': 3},
        'name': 'Germany'
    }

.. bokeh-sampledata-xref:: olympics2014

"""

from __future__ import annotations

import json

from . import package_path

__all__ = ("data",)

with open(package_path("olympics2014.json"), "rb") as f:
    data = json.load(f)
