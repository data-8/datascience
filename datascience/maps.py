"""

Integrates the folium package to draw maps.

"""

from IPython.core.display import HTML
import folium

#########
# Utils #
#########


def map_to_defaults(defaults, params, excludes=[]):
    """
    Squishes together two dictionaries, with one containing defaults
    """
    rval = defaults.copy()
    rval.update(params)
    return {k: v for k, v in rval.items() if k not in excludes}


def hook(f):
    """
    Decorator for obj method, checks for PREFIX_before and PREFIX_after 
    a method is called 
    """

    def call_hook(self, template):
        fn = getattr(self, template % f.__name__, None)
        if callable(fn):
            fn()

    def method(self, *args, **kwargs):
        call_hook(self, 'before_%s')
        rval = f(self, *args, **kwargs)
        call_hook(self, 'after_%s')
        return rval
    
    method.__name__ = getattr(f, '__name__', method.__name__)
    return method


class MapError(Exception):
    
    message = ''
    prefix = 'Please report this to your instructors'
    
    def get_message(self):
        return '%s: %s' % (self.prefix, self.message)


######################
# Folium Abstraction #
######################


def draw_map(center, zoom, **kwargs):
    """ 
    Draw a map with center & zoom containing all points and
    regions that displays points as circles and regions as polygons.
    """

    return Map(
        location=center, 
        zoom_start=zoom, 
        **kwargs).map().to_html()


class MapEntity:

    defaults = {}
    attributes = {} # only data passed to mapper
    data = {}       # includes all data
    dataonly = []   # don't pass these variables to the mapper
    mapper = None   # default function to display this shape on the map
    folium = None   # access to the map object (kind of)

    def __init__(self, **kwargs):
        """ Saves attributes, with defaults """
        self.data = map_to_defaults(self.defaults, kwargs)
        self.attributes = map_to_defaults(self.defaults, kwargs, excludes=self.dataonly)

    def map(self, map=None):
        """ Maps this object, with its own attributes """
        if not callable(self.mapper):
            self.mapper = getattr(map, self.mapper)
        self.folium = self.mapper(**self.attributes)
        return self


class Map(MapEntity):

    mapper = folium.Map
    
    dataonly = ['points', 'regions']

    defaults = {
        'location': [45.5244, -122.6699],   # center -- lat-long pair at the center of the map
        'tiles': 'Stamen Toner',            # zoom -- zoom level
        'zoom_start': 17,                   # points -- a list of MapPoints
        'width': 960,
        'height': 500,
        'points': [],                       # points -- a list of MapPoints
        'regions': []                       # regions -- a list of MapRegions
    }

    @hook
    def to_html(self):
        """
        Takes in a folium map as a parameter and outputs the HTML that
        IPython will display.
        """
        map = self.folium
        map._build_map()
        map_html = map.HTML.replace('"', '&quot;')
        return HTML('<iframe srcdoc="%s" '
                    'style="width: %spx; height: %spx; '
                    'border: none"></iframe>' % (map_html, map.width, map.height))
    
    def before_to_html(self):
        """ called before to_html, add points and regions before outputting map """
        for point in self.data['points']:
            point.map(self.folium)

        for region in self.data['regions']:
            region.map(self.folium)
        

class MapPoint(MapEntity):
    """ A circle. Draw by passing into draw_map."""
    
    mapper = 'circle_marker'
    
    defaults = {
        'location': [],             # location -- lat-long pair at center of circle
        'radius': 10,               # radius -- radius of circle on map
        'popup': '',                # popup -- text that pops up when circle is clicked
        'line_color': '#3186cc',    # line_color -- color of circle border
        'fill_color': '#3186cc',    # fill_color -- color of circle within border
        'fill_opacity': 0.6         # fill_opacity -- opacity of circle fill
    }


class MapRegion(MapEntity):
    """ A polygon. Draw by passing into draw_map."""
    
    mapper = 'geo_json'
    
    defaults = {
        'fill_color': 'blue',
        'fill_opacity': 0.6,
        'line_color': 'black',
        'line_weight': 1,
        'line_opacity': 1
    }