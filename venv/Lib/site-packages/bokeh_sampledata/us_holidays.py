# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Calendar file of US Holidays from Mozilla provided by `icalendar`_.

License `CC BY-SA 3.0`_

Sourced from: https://www.mozilla.org/en-US/projects/calendar/holidays/

This module contains one list: ``us_holidays``.

.. rubric:: ``us_holidays``

.. code-block::

    [
        (datetime.date(1966, 12, 26), 'Kwanzaa'),
        (datetime.date(2000, 1, 1), "New Year's Day"),
        ...
        (datetime.date(2020, 12, 25), 'Christmas Day (US-OPM)')
    ]

.. bokeh-sampledata-xref:: us_holidays

.. _icalendar: https://pypi.org/project/icalendar/

"""

from __future__ import annotations

import icalendar as ic

from . import package_path

__all__ = ("us_holidays",)

with open(package_path("USHolidays.ics"), encoding="utf-8") as f:
    data = ic.Calendar.from_ical(f.read())

us_holidays = sorted((elt.get("dtstart").dt, str(elt.get("summary"))) for elt in data.walk() if elt.name == "VEVENT")
