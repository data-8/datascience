"""

Integrates the folium package to draw maps.

"""

from IPython.core.display import HTML
import json
import folium
import random

#########################
# Exposed Functionality #
#########################


def draw_map(center, **kwargs):
    """ 
    Draw a map with center & zoom containing all points and
    regions that displays points as circles and regions as polygons.
    """

    return Map(
        location=center,
        **kwargs).map().to_html()


#########
# Utils #
#########


def _map_to_defaults(defaults, params, excludes=[], includes=[]):
    """
    Squishes together two dictionaries, with one containing defaults
    """
    rval = defaults.copy()
    rval.update(params)
    return {
        k: v for k, v in rval.items() 
        if k not in excludes
        and k in (includes or rval.keys())
    }


def _hook(f):
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


class ProgrammerError(Exception):
    """ Thrown when programmers make a mistake """
    
    message = ''
    prefix = 'Please report this to your instructors'
    
    def get_message(self):
        return '%s: %s' % (self.prefix, self.message)

class MapError(Exception):
    """ Thrown when users make a mistake """
    pass


######################
# Folium Abstraction #
######################


class MapEntity:

    defaults = {}
    attributes = {} # only data passed to mapper
    data = {}       # includes all data
    dataonly = []   # don't pass these variables to the mapper
    mapper = None   # default function to display this shape on the map
    folium = None   # access to the map object (kind of)

    def __init__(self, **kwargs):
        """ Saves attributes, with defaults """
        self.data = _map_to_defaults(self.defaults, kwargs)
        self.attributes = _map_to_defaults(
            self.defaults, kwargs, excludes=self.dataonly)

    def __setitem__(self, k, v):
        self.data[k] = v

    def __getitem__(self, k):
        return self.data[k]

    @_hook
    def map(self, map=None):
        """ Maps this object, with its own attributes """
        if not callable(self.mapper):
            self.mapper = getattr(map, self.mapper)
        self.folium = self.mapper(**self.attributes)
        return self


class Map(MapEntity):
    """
    Represents a map, potentially with MapPoints and MapRegions

    >>> map = Map().map()
    >>> map.to_html()
    <IPython.core.display.HTML object>
    """

    mapper = folium.Map
    
    dataonly = ['points', 'regions']

    defaults = {
        'location': [45.5244, -122.6699],   # center -- lat-long pair at the center of the map
        'tiles': 'Stamen Toner',            # zoom_start -- zoom level
        'zoom_start': 17,                   # points -- a list of MapPoints
        'width': 960,
        'height': 500,
        'points': [],                       # points -- a list of MapPoints
        'regions': []                       # regions -- a list of MapRegions
    }

    @_hook
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
        """
        Called before to_html, add points and regions before outputting map 
        """
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

    def __init__(self, location, **kwargs):
        super().__init__(location=location, **kwargs)

    def __str__(self):
        return str(list(self.location))


class MapRegion(MapEntity):
    """
    A polygon. Draw by passing into draw_map.

    regions -- list of MapRegions (if present, locations and points are not used)
    locations -- list of list of lat-long pairs for locations with multiple 
                 polygons (if present, points are not used)
    points -- list of lat-long pairs for a normal polygon
    """
    # composite options

    CHILDREN = 'children'  # invoke map() for children
    GROUP = 'group'  # invoke to_json() for children
    ONE = 'one'  # invoke to_polygon() for children
    
    mapper = 'geo_json'

    dataonly = ['points', 'locations', 'regions', 'composite']
    
    defaults = {
        'composite': GROUP,
        'regions': [],
        'locations': [[]],
        'points': [],
        'fill_color': 'blue',
        'fill_opacity': 0.6,
        'line_color': 'black',
        'line_weight': 1,
        'line_opacity': 1
    }

    TEMP_FILE = None  # TODO: temporary workaround - will need a more permanent solution

    allowed_collections = (list, set, tuple)

    def before_map(self):
        """ Prepares for mapping by passing JSON representation """
        self._validate()
        json_str = json.dumps(self._to_json()).replace(']]]}}', ']]]}}\n')
        # self.attributes['geo_str'] = json_str
        if self['locations'][0] or self['points'] or self['regions']:  # TODO: temporary workaround
            import os
            try:
                os.mkdir('cache/')
            except FileExistsError:
                pass
            self.TEMP_FILE = 'cache/'+str(round(random.random()*100))+'temp.json'
            open(self.TEMP_FILE, 'w').write(json_str)
            self.attributes['geo_path'] = self.TEMP_FILE

    def map(self, map=None):
        """
        Takes action, depending on composite
            CHILDREN -- invoke map() for each child
            GROUP -- invoke parent map() normally
            ONE -- invoke parent map() normally
        """
        composite = self['composite']
        if composite in (MapRegion.GROUP, MapRegion.ONE):
            super().map(map)
        elif composite == MapRegion.CHILDREN:
            for region in self['regions']:
                region.map(map)

    def _validate(self):
        """ Checks for validity of data """
        locations = self.data['locations']
        points = self.data['points']

        if not isinstance(locations, self.allowed_collections) \
            or not isinstance(locations[0], self.allowed_collections):
            raise MapError('"Locations" must be a %s of %ss' %
                (' or '.join(self.allowed_collections),
                    's or '.join(self.allowed_collections)))

        if not locations[0]:
            if not isinstance(points, self.allowed_collections):
                raise MapError('"Points" must be a %s' % 
                    ' or '.join(self.allowed_collections))

    @staticmethod
    def to_json(features, **kwargs):
        return _map_to_defaults({
            'type': 'FeatureCollection',
            'features': features
        }, kwargs)

    def _to_json(self):
        """
        Converts MapRegion to folium-ready geoJSON, depending on composite
            CHILDREN -- none
            GROUP -- construct jsons with children
            ONE -- construct polygons with children
        """
        composite = self['composite']
        if composite == MapRegion.GROUP:
            return MapRegion.to_json(
                self._to_feature_json())
        elif composite == MapRegion.ONE:
            return MapRegion.to_json(
                MapRegion.to_feature_json(
                    self._to_polygon_json()))
        return None

    @staticmethod
    def to_feature_json(feature, **kwargs):
        """ Converts single feature into feature JSON """
        return _map_to_defaults({
            'type': 'Feature',
            'properties': {'featurecla': 'Map'},
            'geometry': feature
        }, kwargs)

    def _to_feature_json(self):
        """ Converts to list of feature JSONs """
        return [MapRegion.to_feature_json(feat) 
            for feat in self.to_feature()]

    def to_feature(self):
        """ Converts to list of poylgon JSONs """
        regions = self['regions']

        if regions:
            rval = []
            for region in regions:
                rval += (region.to_feature() or [])
            return rval

        return [self._to_polygon_json()]

    @staticmethod   
    def to_polygon_json(polygon, **kwargs):
        """ Converts single polygon into polygon JSON """
        return _map_to_defaults({
            'type': 'Polygon',
            'coordinates': polygon
        }, kwargs)

    def _to_polygon_json(self):
        """ Converts to one polygon JSON """
        return MapRegion.to_polygon_json(self.to_polygon())

    def to_polygon(self):
        """ Converts to list of polygons, including that of descendants """
        regions, locations, points = self['regions'], self['locations'], self['points']

        if regions:
            rval = []
            for region in regions:
                rval += region.to_polygon()
            return rval

        return self._to_locations()

    def _to_locations(self):
        """ Converts to list of polygons """
        locations, points = self['locations'], self['points']
        if not locations[0]:
            return [points]
        return locations

    def __str__(self):
        return json.dumps(self._to_json())
