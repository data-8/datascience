from IPython.html import widgets # Widget definitions
from IPython.utils.traitlets import Unicode

"""
This file contains the Python backend to draw maps with overlaid polygons.
TODO(sam): Actually make this draw maps
"""

class TestWidget(widgets.DOMWidget):
    _view_name = Unicode('TestWidgetView', sync=True)
    _view_module = Unicode('maps', sync=True)


def draw_map(center, zoom, points=[], regions=[]):
    """Draw a map with center & zoom containing all points and
    regions that displays points as circles and regions as polygons.

    center -- lat-long pair at the center of the map
    zoom -- zoom level
    points -- a sequence of MapPoints
    regions -- a sequence of MapRegions
    """
    # Some magic to get javascript to display the map

class MapPoint:
    """A circle https://developers.google.com/maps/documentation/javascript/shapes#circles"""
    def __init__(self, center, radius, strokeColor, strokeOpacity, strokeWeight, fillColor, fillOpacity):
        pass

class MapRegion:
    """A polygon https://developers.google.com/maps/documentation/javascript/shapes#polygons"""
    def __init__(self, paths, strokeColor, strokeOpacity, strokeWeight, fillColor, fillOpacity):
        """paths -- a list of lat-long pairs or a list of list of lat-long pairs"""
        pass
