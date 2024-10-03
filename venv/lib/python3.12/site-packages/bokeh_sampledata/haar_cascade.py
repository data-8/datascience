# -----------------------------------------------------------------------------
# Copyright (c) Bokeh Contributors
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------
"""Provide a Haar cascade file for face recognition.

License: `MIT license`_

Sourced from the `OpenCV`_ project.

This module contains an attribute ``frontalface_default_path`` . Use this
attribute to obtain the path to a Haar cascade file for frontal face
recognition that can be used by OpenCV.

.. bokeh-sampledata-xref:: haar_cascade

.. _OpenCV: https://opencv.org

"""

from __future__ import annotations

from . import package_path

__all__ = ("frontalface_default_path",)

frontalface_default_path = package_path("haarcascade_frontalface_default.xml")
