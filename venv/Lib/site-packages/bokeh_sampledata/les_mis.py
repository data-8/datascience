# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Provide JSON data for co-occurrence of characters in Les Miserables.

License: `CC BY-ND 4.0`_

Source from http://ftp.cs.stanford.edu/pub/sgb/jean.dat

This module contains one dictionary: ``data``.

.. rubric:: ``data``

.. code-block::

    {
        'nodes': [
            {'name': 'Myriel', 'group': 1},
            ...
            {'name': 'Mme.Hucheloup', 'group': 8}
        ],
        'links': [
            {'source': 1, 'target': 0, 'value': 1},
            ...
            {'source': 76, 'target': 58, 'value': 1}
        ]
    }

.. bokeh-sampledata-xref:: les_mis

"""

from __future__ import annotations

import json

from . import package_path

__all__ = ("data",)

with open(package_path("les_mis.json"), "rb") as f:
    data = json.load(f)
