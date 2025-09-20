# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Historical and projected population data by age, gender, and country.

License: `CC BY 3.0 IGO`_

Sourced from: https://population.un.org/wpp/Download/Standard/Population/

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.population.data`

.. bokeh-sampledata-xref:: population

"""

from __future__ import annotations

from . import package_csv

__all__ = ("data",)

data = package_csv("WPP2012_SA_DB03_POPULATION_QUINQUENNIAL.csv", encoding="CP1250")
data = data[data.Sex != "Both"]
data = data.drop(["VarID", "Variant", "MidPeriod", "SexID", "AgeGrpSpan"], axis=1)
data = data.rename(columns={"Time": "Year"})
data.Value *= 1000
