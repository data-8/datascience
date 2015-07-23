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
    points -- a list of MapPoints
    regions -- a list of MapRegions

    TODO(sam): Add regions functionality
    """
    map = folium.Map(
        location=center,
        tiles='Stamen Toner',
        zoom_start=zoom,
        width=MAP_WIDTH,
        height=MAP_HEIGHT
    )

    for point in points:
        map.circle_marker(**point.attributes)

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
    """A circle. Draw by passing into draw_map."""

    def __init__(self, location, radius=10, popup='', line_color='#3186cc', fill_color='#3186cc', fill_opacity=0.6):
        """
        location -- lat-long pair at center of circle
        radius -- radius of circle on map
        popup -- text that pops up when circle is clicked
        line_color -- color of circle border
        fill_color -- color of circle within border
        fill_opacity -- opacity of circle fill
        """
        self.attributes = {
            'location': location,
            'radius': radius,
            'popup': popup,
            'line_color': line_color,
            'fill_color': fill_color,
            'fill_opacity': fill_opacity
        }

class MapRegion:
    """A polygon. Draw by passing into draw_map."""
    def __init__(self, paths, strokeColor, strokeOpacity, strokeWeight, fillColor, fillOpacity):
        """paths -- a list of lat-long pairs or a list of list of lat-long pairs"""
        pass
