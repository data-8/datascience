"""Integrates the folium package to draw maps."""

from IPython.core.display import HTML
import folium

MAP_WIDTH = 960
MAP_HEIGHT = 500

def draw_map(center, zoom=17, points=[], regions=[]):
    """Draw a map with center & zoom containing all points and
    regions that displays points as circles and regions as polygons.

    center -- lat-long pair at the center of the map
    zoom -- zoom level
    points -- a sequence of MapPoints
    regions -- a sequence of MapRegions

    TODO(sam): Add points and regions functionality
    """
    map = folium.Map(
        location=center,
        tiles='Stamen Toner',
        zoom_start=zoom,
        width=MAP_WIDTH,
        height=MAP_HEIGHT
    )

    return _to_html(map)

def _to_html(map):
    """Takes in a folium map as a parameter and outputs the HTML that
    IPython will display."""
    map._build_map()
    map_html = map.HTML.replace('"', '&quot;')
    return HTML('<iframe srcdoc="%s" '
                'style="width: %spx; height: %spx; '
                'border: none"></iframe>' % (map_html, map.width, map.height))

class MapPoint:
    """A circle https://developers.google.com/maps/documentation/javascript/shapes#circles"""
    def __init__(self, center, radius, strokeColor, strokeOpacity, strokeWeight, fillColor, fillOpacity):
        pass

class MapRegion:
    """A polygon https://developers.google.com/maps/documentation/javascript/shapes#polygons"""
    def __init__(self, paths, strokeColor, strokeOpacity, strokeWeight, fillColor, fillOpacity):
        """paths -- a list of lat-long pairs or a list of list of lat-long pairs"""
        pass
