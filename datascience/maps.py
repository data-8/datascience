"""Draw maps using folium."""

from IPython.core.display import HTML

import folium
import pandas

import collections
import collections.abc
import json
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


def _lat_lons_from_geojson(s):
    """Return a latitude-longitude pairs from nested GeoJSON coordinates.

    GeoJSON coordinates are always stored in (longitude, latitude) order.
    """
    if len(s) >= 2 and isinstance(s[0], (int, float)) and isinstance(s[0], (int, float)):
        lat, lon = s[1], s[0]
        return [(lat, lon)]
    else:
        return [lat_lon for sub in s for lat_lon in _lat_lons_from_geojson(sub)]


def load_geojson(path_or_json_or_string):
    """Load a geoJSON string, object, or file. Return a dict of features keyed by ID."""
    assert path_or_json_or_string
    data = None
    if isinstance(path_or_json_or_string, (dict, list)):
        data = path_or_json_or_string
    try:
        data = json.loads(path_or_json_or_string)
    except ValueError:
        pass
    try:
        data = json.loads(open(path_or_json_or_string, 'r').read())
    except FileNotFoundError:
        pass
    if not data:
        raise MapError('MapData accepts a valid geoJSON object, '
            'geoJSON string, or path to a geoJSON file')
    return Map(_read_geojson_features(data))


def _read_geojson_features(data, features=None, prefix=""):
    """Return a dict of features keyed by ID."""
    if features is None:
        features = collections.OrderedDict()
    for i, feature in enumerate(data['features']):
        key = feature.get('id', prefix + str(i))
        feature_type = feature['geometry']['type']
        if feature_type == 'FeatureCollection':
            _read_geojson_features(feature, features, prefix + '.' + key)
        elif feature_type == 'Point':
            value = _convert_point(feature)
        else:
            value = Region(feature)
        features[key] = value
    return features


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


class Map(_FoliumWrapper, collections.abc.Mapping):
    """A map from IDs to features. Keyword args are forwarded to folium."""

    _mapper = folium.Map
    _default_lat_lon = (37.872, -122.258)
    _default_zoom = 12

    def __init__(self, features=(), ids=(), width=960, height=500, **kwargs):
        if isinstance(features, (tuple, set, list)):
            if len(ids) == len(features):
                features = dict(zip(ids, features))
            else:
                assert len(ids) == 0
                features = dict(enumerate(features))
        assert isinstance(features, dict)
        self._features = features
        self._attrs = {
            'tiles': 'Stamen Toner',
            'max_zoom': 17,
            'min_zoom': 10,
        }
        self._width = width
        self._height = height
        self._attrs.update(kwargs)

    def __getitem__(self, id):
        return self._features[id]

    def __len__(self):
        return len(self._features)

    def __iter__(self):
        return iter(self._features)

    def _set_folium_map(self):
        attrs = {'width': self._width, 'height': self._height}
        attrs.update(self._autozoom())
        attrs.update(self._attrs.copy())
        # Enforce zoom consistency
        attrs['max_zoom'] = max(attrs['zoom_start']+2, attrs['max_zoom'])
        attrs['min_zoom'] = min(attrs['zoom_start']-2, attrs['min_zoom'])
        self._folium_map = self._mapper(**attrs)
        for feature in self._features.values():
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

        lat_lons = [lat_lon for feature in self._features.values() for
                    lat_lon in feature.lat_lons]
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

    def geojson(self, feature_id):
        """Return GeoJSON."""
        return {'id': feature_id}


class Marker(_MapFeature):
    """A marker displayed with Folium's simple_marker method.

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
        assert isinstance(lat, (int, float))
        assert isinstance(lon, (int, float))
        self.lat_lon = (lat, lon)
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
        return MapPoint(self.lat_lon[:], **self._attrs)

    @property
    def _folium_kwargs(self):
        attrs = self._attrs.copy()
        attrs['location'] = self.lat_lon
        return attrs

    def geojson(self, feature_id):
        """GeoJSON representation of the point."""
        lat, lon = self.lat_lon
        return {
            'type': 'Feature',
            'id': feature_id,
            'geometry': {
                'type': 'Point',
                'coordinates': (lon, lat),
            },
        }

    def format(self, **kwargs):
        attrs = self._attrs.copy()
        attrs.update(kwargs)
        lat, lon = self.lat_lon
        return Marker(lat, lon, **attrs)


def as_markers(latitudes, longitudes, labels=None, **kwargs):
    """Return a list of Markers from columns of coordinates and labels."""
    assert len(latitudes) == len(longitudes)
    if labels is None:
        return [Marker(lat, lon, **kwargs) for lat, lon in zip(latitudes, longitudes)]
    else:
        assert len(labels) == len(latitudes)
        return [Marker(lat, lon, popup, **kwargs) for
                lat, lon, popup in zip(latitudes, longitudes, labels)]


class Region(_MapFeature):
    """A GeoJSON feature displayed with Folium's geo_json method."""

    _map_method_name = 'geo_json'

    def __init__(self, geojson, **kwargs):
        assert 'type' in geojson
        assert geojson['type'] == 'Feature'
        self._geojson = geojson
        self._attrs = kwargs

    @property
    def lat_lons(self):
        return _lat_lons_from_geojson(self._geojson['geometry']['coordinates'])

    def copy(self):
        """Return a deep copy"""
        return Region(self._geojson.copy(), **self._attrs)

    @property
    def _folium_kwargs(self):
        attrs = self._attrs.copy()
        attrs['geo_str'] = json.dumps(self._geojson)
        return attrs

    def geojson(self, feature_id):
        if self._geojson.get('id', feature_id) == feature_id:
            return self._geojson
        else:
            geo = self._geojson.copy()
            geo['id'] = feature_id
            return geo

    def format(self, **kwargs):
        attrs = self._attrs.copy()
        attrs.update(kwargs)
        return Region(self._geojson, **attrs)
