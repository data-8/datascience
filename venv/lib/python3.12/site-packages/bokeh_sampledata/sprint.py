# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Historical results for Olympic sprints by year.

Sourced from public news sources.

This module contains one pandas Dataframe: ``sprint``.

.. rubric:: ``sprint``

:bokeh-dataframe:`bokeh.sampledata.sprint.sprint`

.. bokeh-sampledata-xref:: sprint

"""

from __future__ import annotations

from . import package_csv

__all__ = ("sprint",)

sprint = package_csv(
    "sprint.csv",
    skipinitialspace=True,
    escapechar="\\",
)
