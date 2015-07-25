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


def draw_map(center=None, **kwargs):
    """ 
    Draw a map with center & zoom containing all points and
    regions that displays points as circles and regions as polygons.
    """
    return get_map(center, **kwargs).to_html()


def get_map(center=None, **kwargs):
    """
    Get a map and play with it before drawing it. Use to_html() to draw.
    """
    return Map(location=center, **kwargs).map()


class Data:
    """
    Abstracts away a geoJSON string, object, or file

    >>> data = Data('../data/us-states.json')
    >>> data.CA
    (list of coordinates)
    """

    _data = None

    def __init__(self, path_or_json_or_string):
        """ Loads geoJSON """
        jsons = path = string = path_or_json_or_string
        if isinstance(jsons, (dict, list)):
            self._data = jsons
        try:
            self._data = json.loads(string)
        except ValueError:
            pass
        try:
            self._data = json.loads(open(path, 'r').read())
        except FileNotFoundError:
            pass
        if not self._data:
            raise MapError('Data accepts a valid geoJSON object,\
            geoJSON string, or path to a geoJSON file')
        self.process_data()

    def process_data(self):
        """
        Processes the data, and makes each list coordinates accessible via
        the ID of the feature.
        """
        self._features = self._data['features']
        for i, feature in enumerate(self._features):
            setattr(self, feature.get('id', i),
                    MapRegion(locations=feature['geometry']['coordinates'],
                              geo_formatted=True))

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

    def call_hook(self, template, *args, **kwargs):
        fn = getattr(self, template % f.__name__, None)
        if callable(fn):
            fn(*args, **kwargs)

    def method(self, *args, **kwargs):
        call_hook(self, '_before_%s', *args, **kwargs)
        rval = f(self, *args, **kwargs)
        call_hook(self, '_after_%s', *args, **kwargs)
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
    """
    General Map entity
    - attributes get and set, using map[prop]
    - all other data get and set, using map.prop
    """

    _init = {}                                  # params for initial call
    _defaults = {}                              # defaults for both attributes and data
    _attributes = None                          # only data passed to mapper
    _allowed_attributes = []                    # allowed attributes
    _mapper = None                              # default function to display this shape on the map
    _folium = None                              # access to the map object (kind of)
    _allowed_collections = (list, set, tuple)   # allowed collections for list of locations

    @_hook
    def __init__(self, **kwargs):
        """ Saves attributes, with defaults """
        self._init, self._attributes = kwargs, {}
        self._attributes = _map_to_defaults(
            self._defaults, kwargs, includes=self._allowed_attributes)
        self.set(**_map_to_defaults(self._defaults, kwargs))

    def __setitem__(self, k, v):
        """ Set attribute """
        self._attributes[k] = v

    def __getitem__(self, k):
        """ Get attribute """
        return self._attributes[k]
    
    def __delitem__(self, k):
        """ Delete attribute """
        del self._attributes[k]

    def set(self, override=False, **kwargs):
        """ Transform args into properties """
        for k, v in kwargs.items():
            attr = getattr(self, k, None)
            if callable(attr) and not override:
                raise ProgrammerError('"%s" is an existing method' % k)
            setattr(self, k, v)
        return self
    
    def get(self, *args):
        """ Get a set of variables - use pairs to set getattr default """
        get = lambda val: getattr(self, val[0], val[1]) \
            if isinstance(val, tuple) else getattr(self, val)
        return (get(var) for var in args)

    @_hook
    def map(self, map=None):
        """ Maps this object, with its own attributes """
        if not callable(self._mapper):
            self._mapper = getattr(map, self._mapper)
        self._folium = self._mapper(**self._attributes)
        return self


class Map(MapEntity):
    """
    Represents a map, potentially with MapPoints and MapRegions
    - For description of all capabilities, see Folium Map:
      https://github.com/python-visualization/folium/blob/master/folium/folium.py#L58
      
    points -- a list of MapPoints
    regions -- a list of MapRegions

    >>> map = Map().map()
    >>> map.to_html()
    <IPython.core.display.HTML object>
    """

    _mapper = folium.Map
    
    _allowed_attributes = [
        'location',     # location or center -- lat-long pair at the center of the map
        'width',        # width -- pixel int or percentage string
        'height',       # height -- pixel int or percentage string
        'tiles',        # tiles -- map tileset, or "theme"
        'API_key',      # API_key -- for Cloudmade or Mapbox tiles
        'max_zoom',     # max_zoom -- maximum zoom, default 18
        'zoom_start',   # zoom_start -- starting zoom level
    ]

    _defaults = {
        'location': [45.5244, -122.6699],
        'tiles': 'Stamen Toner',
        'zoom_start': 17,
        'width': 960,
        'height': 500,
        'points': [],
        'regions': []
    }

    @_hook
    def to_html(self):
        """
        Takes in a folium map as a parameter and outputs the HTML that
        IPython will display.
        """
        map = self._folium
        map._build_map()
        map_html = map.HTML.replace('"', '&quot;')
        return HTML('<iframe srcdoc="%s" '
                    'style="width: %spx; height: %spx; '
                    'border: none"></iframe>' % (map_html, map.width, map.height))

    def _before_to_html(self, *args, **kwargs):
        """
        Called before to_html, add points and regions before outputting map 
        """
        for point in self.points:
            point.map(self._folium)

        for region in self.regions:
            region.map(self._folium)

        # uncomment the following with new Folium release
        # if 'zoom_start' not in self._init:
        #     self._autofit(self.bounds)
    
    def _before_map(self, *args, **kwargs):
        """ Autozoom map where appropriate """
        if self.location:
            self._dynamic_arg('location')
            
        self.bounds = self._autobounds()
        
        if not self.location:
            self._autocenter(self.bounds)

        # remove with new Folium release
        if 'zoom_start' not in self._init:
            self._autofit(self.bounds)
            
    def _autobounds(self):
        """ Simple calculation for bounds """
        bounds = {}

        def check(prop, compare, extreme, val):
            opp = min if compare is max else max
            bounds.setdefault(prop, val)
            bounds[prop] = opp(compare(bounds[prop], val), extreme)

        def bound_check(coord):
            if not coord:
                return
            if isinstance(coord, self._allowed_collections):
                if isinstance(coord[0], (int, float)):
                    long, lat = coord
                    check('max_lat', max, 90, lat)
                    check('min_lat', min, -90, lat)
                    check('max_long', max, 180, long)
                    check('min_long', min, -180, long)
                else:
                    [bound_check(c) for c in coord]
            else:
                raise ProgrammerError('Ended up with %s.' % type(coord))

        bound_check([point.location for point in self.points])
        bound_check([region.to_polygon() for region in self.regions])
        return bounds
        
    def _autofit(self, bounds):
        """ Automatically fits everything with the maximum zoom possible """
        if not bounds:
            return
        
        # uncomment with new Folium release
        # self._folium.fit_bounds(
        #     [bounds['min_long'], bounds['min_lat']],
        #     [bounds['max_long'], bounds['max_lat']]
        # )
        
        # remove the following with new Folium release
        # rough approximation, assuming max_zoom is 18
        import math
        try:
            factor = 1.2
            width = ((bounds['max_lat'] - bounds['min_lat']) or 2)
            height = ((bounds['max_long'] - bounds['min_long']) or 2)
            area, max_area = width*height, 180*360
            zoom = math.log(area/max_area)/-factor
            self['zoom_start'] = max(1, min(18, round(zoom)))
        except ValueError:
            raise MapError('Uh oh. Is the data formatted as long-lat \
            pairs? (Check. If the smallest number is less than -90, \
            those are long-lat pairs) If so, please add geo_formatted=True \
            to create_map')

    def _autocenter(self, bounds):
        """ Find the center """
        if not bounds:
            return
        
        midpoint = lambda a, b: (a + b)/2
        self['location'] = (
            midpoint(bounds['min_lat'], bounds['max_lat']),
            midpoint(bounds['min_long'], bounds['max_long'])
        )
        
    def _dynamic_arg(self, key):
        """ Checks the first arg in check_map for other options """
        val = getattr(self, key, None)
        if val:
            setattr(self, key, None)
            del self[key]
            if isinstance(val, MapRegion):
                self.regions.append(val)
            elif isinstance(val, MapPoint):
                self.points.append(val)
            else:
                setattr(self, key, val)
                self[key] = val

    @classmethod
    def reverse(cls, lst_or_coord_or_obj):
        """
        Reverses all coordinates in a list or list of lists or so on
        and so forth - effectively translates all EPSG-compliant coordinates
        into valid geoJSON coordinates
        """
        lst = coord = obj = lst_or_coord_or_obj
        if not lst:
            return
        if isinstance(coord[0], (int, float)):
            return coord[::-1]
        if isinstance(lst, cls._allowed_collections):
            return [cls.reverse(item) for item in lst]
        return obj


class MapPoint(MapEntity):
    """ A circle. Draw by passing into draw_map."""
    
    _mapper = 'circle_marker'
    
    _allowed_attributes = [
        'location',         # location -- lat-long pair at center of circle
        'radius',           # radius -- radius of circle on map
        'popup',            # popup -- text that pops up when circle is clicked
        'line_color',       # line_color -- color of circle border
        'fill_color',       # fill_color -- color of circle within border
        'fill_opacity'      # fill_opacity -- opacity of circle fill
    ]
    
    _defaults = {
        'location': [],
        'radius': 10,
        'popup': '',
        'line_color': '#3186cc',
        'fill_color': '#3186cc',
        'fill_opacity': 0.6
    }

    def __init__(self, location_or_x, y=None, **kwargs):
        location = x = location_or_x
        if isinstance(location, self._allowed_collections):
            pass
        elif isinstance(x, (int, float)):
            location = (x, y)
        super().__init__(location=location, **kwargs)

    def __str__(self):
        return str(list(self.location))

    def _after___init__(self, *args, **kwargs):
        """ Translates all lat-long pairs into long-lat pairs, from EPSG-compliant to GeoJSON format """
        if 'geo_formatted' not in kwargs or not kwargs['geo_formatted']:
            self.location = Map.reverse(self.location)


class MapRegion(MapEntity):
    """
    A polygon. Draw by passing into draw_map.

    regions -- list of MapRegions (if present, locations and points are not used)
    locations -- list of list of lat-long pairs for locations with multiple
                 polygons (if present, points are not used)
    points -- list of lat-long pairs for a normal polygon
    geo_formatted -- boolean, True if the data passed in is formatted as long-lat pairs
    
    Note: GeoJSON requires long-lat-alt triplets, or just long-lat pairs, although EPSG:4326
    states that the coordinate order should be lat-long. For this reason, MapRegion has to
    flip incoming lat-long pairs to become long-lat pairs. By default, MapRegion accepts
    lat-long pairs, to match MapPoint's behavior.
    """
    
    # composite options

    CHILDREN = 'children'  # invoke map() for children
    GROUP = 'group'  # invoke to_json() for children
    ONE = 'one'  # invoke to_polygon() for children
    
    _mapper = 'geo_json'

    _allowed_attributes = [
        'geo_path',
        'geo_str',
        'data_out',
        'data',
        'columns',
        'key_on',
        'threshold_scale',
        'fill_color',
        'fill_opacity',
        'line_color',
        'line_weight',
        'line_opacity',
        'legend_name',
        'topojson',
        'reset'
    ]
    
    _defaults = {
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

    CACHE_DIR = 'cache'
    TEMP_FILE = None  # TODO: temporary workaround - will need a more permanent solution

    def _after___init__(self, *args, **kwargs):
        """ Translates all lat-long pairs into long-lat pairs, from EPSG-compliant to GeoJSON format """
        if 'geo_formatted' not in kwargs or not kwargs['geo_formatted']:
            if 'locations' in kwargs:
                self.locations = Map.reverse(kwargs['locations'])
            if 'points' in kwargs:
                self.points = Map.reverse(kwargs['points'])

    def _before_map(self, *args, **kwargs):
        """ Prepares for mapping by passing JSON representation """
        self._validate()
        json_str = json.dumps(self._to_json()).replace(']]]}}', ']]]}}\n')
        # self.attributes['geo_str'] = json_str
        if self.locations[0] or self.points or self.regions:  # TODO: temporary workaround
            import os
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            self.TEMP_FILE = os.path.join(
                self.CACHE_DIR, str(round(random.random()*1000))+'temp.json')
            open(self.TEMP_FILE, 'w').write(json_str)
            self._attributes['geo_path'] = self.TEMP_FILE

    def map(self, map=None):
        """
        Takes action, depending on composite
            CHILDREN -- invoke map() for each child
            GROUP -- invoke parent map() normally
            ONE -- invoke parent map() normally
        """
        composite = self.composite
        if composite in (MapRegion.GROUP, MapRegion.ONE):
            super().map(map)
        elif composite == MapRegion.CHILDREN:
            for region in self.regions:
                region.map(map)

    def _validate(self):
        """ Checks for validity of data """
        locations, points = self.get('locations', 'points')

        if not isinstance(locations, self._allowed_collections) \
            or not isinstance(locations[0], self._allowed_collections):
            raise MapError('"Locations" must be a %s of %ss' %
                (' or '.join(self._allowed_collections),
                    's or '.join(self._allowed_collections)))

        if not locations[0]:
            if not isinstance(points, self._allowed_collections):
                raise MapError('"Points" must be a %s' % 
                    ' or '.join(self._allowed_collections))

    @staticmethod
    def to_json(features, **kwargs):
        """ Converts any features list to FeatureCollection JSON """
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
        composite = self.composite
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
        regions = self.regions

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
        regions, locations, points = self.get('regions', 'locations', 'points')

        if regions:
            rval = []
            for region in regions:
                rval += region.to_polygon()
            return rval

        return self._to_locations()

    def _to_locations(self):
        """ Converts to list of polygons """
        locations, points = self.get('locations', 'points')
        if not locations[0]:
            return [points]
        return locations

    def __str__(self):
        return json.dumps(self._to_json())