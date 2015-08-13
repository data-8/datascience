"""Draw maps using folium."""

from IPython.core.display import HTML
import json
import folium
import functools
import random


def _inline_map(m, width, height):
    """Returns an embedded iframe of a folium.map."""
    m._build_map()
    src = m.HTML.replace('"', '&quot;')
    style = "width: {}px; height: {}px".format(width, height)
    return '<iframe srcdoc="{}" style="{}"; border: none"></iframe>'.format(src, style)


def _convert_point(feature):
    """Convert a GeoJSON point to a Marker."""
    lon, lat = feature['geometry']['coordinates']
    popup = feature['properties'].get('name', '')
    return Marker(lat, lon)


def load_geojson(path_or_json_or_string):
    """Load a geoJSON string, object, or file. Return a dict of map features."""
    assert path_or_json_or_string
    data = None
    if isinstance(path_or_json_or_string, (dict, list)):
        data = path_or_json_or_string
    elif isinstance(path_or_json_or_string, str):
        try:
            data = json.loads(path_or_json_or_string)
        except ValueError:
            pass
        try:
            data = json.loads(open(path_or_json_or_string, 'r').read())
        except FileNotFoundError, IOError:
            pass
        # TODO Handle web address
    if not data:
        raise MapError('MapData accepts a valid geoJSON object, '
            'geoJSON string, or path to a geoJSON file')

    return Region(data)


class _FoliumWrapper:
    """A map element that can be drawn."""
    _width = 180
    _height = 180
    _folium_map = None

    def draw(self):
        """Return a Folium rendering."""
        raise NotImplementedError

    def draw(self):
        if not self._folium_map:
            self._set_folium_map()
        return self._folium_map

    def _set_folium_map(self):
        """Set the _folium_map attribute to a map."""

    def as_html(self):
        if not self._folium_map:
            self.draw()
        return _inline_map(self._folium_map, self._width, self._height)

    def show(self):
        return HTML(self.as_html(self._width, self._height))

    def _repr_html_(self):
        return self.as_html()


class Map(_FoliumWrapper):
    """A map displaying features. Keyword args are forwarded to folium."""

    _mapper = folium.Map
    _default_lat_lon = (37.872, -122.258)
    _default_zoom = 12

    def __init__(self, features=(), width=960, height=500, **kwargs):
        self._features = features
        self._attrs = {
            'tiles': 'Stamen Toner',
            'max_zoom': 17,
            'min_zoom': 10,
        }
        self._width = width
        self._height = height
        self._attrs.update(kwargs)

    def _set_folium_map(self):
        attrs = {'width': self._width, 'height': self._height}
        attrs.update(self._autozoom())
        attrs.update(self._attrs.copy())
        # Enforce zoom consistency
        attrs['max_zoom'] = max(attrs['zoom_start'], attrs['max_zoom'])
        attrs['min_zoom'] = min(attrs['zoom_start'], attrs['min_zoom'])
        self._folium_map = self._mapper(**attrs)
        for feature in self._features:
            feature.draw_on(self._folium_map)

    def _autozoom(self):
        """Simple calculation for bounds."""
        bounds = {}
        attrs = {}

        def check(prop, compare, extreme, val):
            opp = min if compare is max else max
            bounds.setdefault(prop, val)
            bounds[prop] = opp(compare(bounds[prop], val), extreme)

        def bound_check(lat_lon):
            lat, lon = lat_lon
            check('max_lat', max, 90, lat)
            check('min_lat', min, -90, lat)
            check('max_lon', max, 180, lon)
            check('min_lon', min, -180, lon)

        lat_lons = [lat_lon for f in self._features for lat_lon in f.lat_lons]
        if not lat_lons:
            lat_lons.append(self._default_lat_lon)
        for lat_lon in lat_lons:
            bound_check(lat_lon)

        midpoint = lambda a, b: (a + b)/2
        attrs['location'] = (
            midpoint(bounds['min_lat'], bounds['max_lat']),
            midpoint(bounds['min_lon'], bounds['max_lon'])
        )

        # TODO(Alvin): uncomment with new Folium release
        # self._folium_map.fit_bounds(
        #     [bounds['min_long'], bounds['min_lat']],
        #     [bounds['max_long'], bounds['max_lat']]
        # )

        # remove the following with new Folium release
        # rough approximation, assuming max_zoom is 18
        import math
        try:
            factor = 1.2
            lat_diff = bounds['max_lat'] - bounds['min_lat']
            lon_diff = bounds['max_lon'] - bounds['min_lon']
            area, max_area = lat_diff*lon_diff, 180*360
            if area:
                zoom = math.log(area/max_area)/-factor
            else:
                zoom = self._default_zoom
            zoom = max(1, min(18, round(zoom)))
            attrs['zoom_start'] = zoom
        except ValueError as e:
            raise Exception('Check that your locations are lat-lon pairs', e)

        return attrs


class _MapFeature(_FoliumWrapper):
    """A feature displayed on a map."""

    _map_method_name = ""
    _attrs = {}

    @property
    def lat_lons(self):
        """All lat_lons that describe a map feature (for zooming)."""
        return []

    def draw_on(self, folium_map):
        """Add feature to Folium map object."""
        f = getattr(folium_map, self._map_method_name)
        f(**self._folium_kwargs)

    def _set_folium_map(self):
        """A map containing only the feature."""
        m = Map(features=[self], width=self._width, height=self._height)
        self._folium_map = m.draw()

    @property
    def _folium_kwargs(self):
        """kwargs for a call to a map method."""
        return self._attrs.copy()


class Marker(_MapFeature):
    """A marker wrapping Folium's simple_marker method.

    location -- lat-lon pair
    popup -- text that pops up when marker is clicked

    Defaults from Folium:
    marker_color='blue'
    marker_icon='info-sign'
    clustered_marker=False
    icon_angle=0
    width=300
    """

    _map_method_name = 'simple_marker'

    def __init__(self, lat, lon, popup="", **kwargs):
        self.lat_lon = (lat, lon)
        if isinstance(popup, str):
            self._name = popup
        self._attrs = {
            'popup': popup,
            'popup_on': bool(popup),
        }
        self._attrs.update(kwargs)

    @property
    def lat_lons(self):
        return [self.lat_lon]

    def copy(self):
        """Return a deep copy"""
        return MapPoint(self.lat_lon[:], **self._attrs.copy())

    @property
    def _folium_kwargs(self):
        attrs = self._attrs.copy()
        attrs['location'] = self.lat_lon
        return attrs

    @property
    def geojson(self):
        """GeoJSON representation of the point."""


class Region(MapFeature):
    """An arbitrary set of features wrapping Folium's geo_json method."""

    _map_method_name = 'geo_json'

    def _before_map(self, *args, **kwargs):
        """Prepares for mapping by passing JSON representation"""
        self._validate()
        json_str = json.dumps(self._to_json()).replace(']]]}}', ']]]}}\n')

        # self.attributes['geo_str'] = json_str

        # TODO(Alvin): find soln to temporary workaround
        if self.locations[0] or self.lat_lons or self.regions:
            import os
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            self.TEMP_FILE = os.path.join(
                self.CACHE_DIR, str(round(random.random()*1000))+'temp.json')
            open(self.TEMP_FILE, 'w').write(json_str)
            self._attributes['geo_path'] = self.TEMP_FILE

    def map(self, map=None):
        """Takes action, depending on composite

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
        """Checks for validity of data"""
        locations, lat_lons = self.get('locations', 'lat_lons')

        if not isinstance(lat_lons, self._allowed_collections):
            raise MapError('"Points" must be a %s' %
                ' or '.join(self._allowed_collections))

    @staticmethod
    def to_json(features, **kwargs):
        """Converts any features list to FeatureCollection JSON"""
        return _map_to_defaults({
            'type': 'FeatureCollection',
            'features': features
        }, kwargs)

    def _to_json(self):
        """Converts MapRegion to folium-ready geoJSON, depending on composite.

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
        """Converts single feature into feature JSON"""
        return _map_to_defaults({
            'type': 'Feature',
            'properties': {'featurecla': 'Map'},
            'geometry': feature
        }, kwargs)

    def _to_feature_json(self):
        """Converts to list of feature JSONs"""
        return [MapRegion.to_feature_json(feat)
            for feat in self.to_feature()]

    def to_feature(self):
        """Converts to list of poylgon JSONs"""
        regions = self.regions

        if regions:
            rval = []
            for region in regions:
                rval += (region.to_feature() or [])
            return rval

        return [self._to_polygon_json()]

    @staticmethod
    def to_polygon_json(polygon, **kwargs):
        """Converts single polygon into polygon JSON"""
        return _map_to_defaults({
            'type': 'Polygon',
            'coordinates': polygon
        }, kwargs)

    def _to_polygon_json(self):
        """Converts to one polygon JSON"""
        return MapRegion.to_polygon_json(self.to_polygon())

    def to_polygon(self):
        """Converts to list of polygons, including that of descendants"""
        regions, locations, lat_lons = self.get('regions', 'locations', 'lat_lons')

        if regions:
            rval = []
            for region in regions:
                rval += region.to_polygon()
            return rval

        return self._to_locations()

    def _to_locations(self):
        """Converts to list of polygons"""
        locations, lat_lons = self.get('locations', 'lat_lons')
        if not locations[0]:
            return [lat_lons]
        return locations

    def __str__(self):
        return json.dumps(self._to_json())

    def copy(self):
        """Makes a deep copy of MapRegion"""
        attributes = {k: v for k, v in self._attributes.items()
                      if k not in ['locations', 'regions', 'lat_lons']}
        return MapRegion(
            regions=[region.copy() for region in self.regions],
            locations=[loc.copy() for loc in self.locations],
            lat_lons=[pt[:] for pt in self.lat_lons],
            geo_formatted=True,
            **attributes)
