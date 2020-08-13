"""Draw maps using folium."""

__all__ = ['Map', 'Marker', 'Circle', 'Region', 'get_coordinates']


import IPython.display
import itertools
import folium
from folium.plugins import MarkerCluster, BeautifyIcon
import pandas
import numpy as np
import matplotlib as mpl
import pkg_resources
import branca.colormap as cm

import abc
import collections
import collections.abc
import functools
import json
import math
import random

from .tables import Table

_number = (int, float, np.number)


class _FoliumWrapper(abc.ABC):
    """A map element that can be drawn."""
    _width = 0
    _height = 0
    _folium_map = None

    def draw(self):
        """Draw and cache map."""
        if not self._folium_map:
            self._set_folium_map()
        return self._folium_map

    def as_html(self):
        """Generate HTML to display map."""
        if not self._folium_map:
            self.draw()
        return self._inline_map(self._folium_map, self._width, self._height)

    def show(self):
        """Publish HTML."""
        IPython.display.display(IPython.display.HTML(self.as_html()))

    def _repr_html_(self):
        return self.as_html()

    @staticmethod
    def _inline_map(m, width, height):
        """Returns an embedded iframe of a folium.map."""
        html = m._repr_html_()
        return html

    @abc.abstractmethod
    def _set_folium_map(self):
        """Set the _folium_map attribute to a map."""


class Map(_FoliumWrapper, collections.abc.Mapping):
    """A map from IDs to features. Keyword args are forwarded to folium."""

    _mapper = folium.Map
    _default_lat_lon = (37.872, -122.258)
    _default_zoom = 12

    def __init__(self, features=(), ids=(), width=960, height=500, **kwargs):
        if isinstance(features, np.ndarray):
            features = list(features)
        if isinstance(features, collections.abc.Sequence):
            if len(ids) == len(features):
                features = dict(zip(ids, features))
            else:
                assert len(ids) == 0
                features = dict(enumerate(features))
        elif isinstance(features, _MapFeature):
            features = {0: features}
        assert isinstance(features, dict), 'Map takes a list or dict of features'
        tile_style = None
        if "tiles" in kwargs:
            tile_style = kwargs.pop("tiles")
        self._index_map = self._cluster_labels = self._colorbar_scale = None
        self._radius_in_meters = False
        if "index_map" in kwargs:
            self._index_map = kwargs.pop("index_map")
        if "cluster_labels" in kwargs:
            self._cluster_labels = kwargs.pop("cluster_labels")
        if "colorbar_scale" in kwargs:
            self._colorbar_scale = kwargs.pop("colorbar_scale")
        if "include_color_scale_outliers" in kwargs:
            self._include_color_scale_outliers = kwargs.pop("include_color_scale_outliers")
        if "radius_in_meters" in kwargs and kwargs["radius_in_meters"] is not None:
            self._radius_in_meters = kwargs.pop("radius_in_meters")
        self._features = features
        self._attrs = {
            'tiles': tile_style if tile_style else 'OpenStreetMap',
            'max_zoom': 17,
            'min_zoom': 10,
        }
        self._width = width
        self._height = height
        self._attrs.update(kwargs)

    def copy(self):
        """
        Copies the current Map into a new one and returns it.
        """

        m = Map(features=self._features, width=self._width,
                   height=self._height, **self._attrs)
        m._folium_map = self._folium_map
        return m

    def __getitem__(self, id):
        return self._features[id]

    def __len__(self):
        return len(self._features)

    def __iter__(self):
        return iter(self._features)

    def _set_folium_map(self):
        self._folium_map = self._create_map()
        if 'clustered_marker' in self._attrs and self._attrs['clustered_marker']:
            def customize_marker_cluster(color, label):
                # Returns string for icon_create_function
                hexcolor = mpl.colors.to_hex(color)
                return f"""
                    function(cluster) {{ 
                        return L.divIcon({{ 
                            html: `<div
                              style='
                                opacity: 0.85; 
                                background-color: {hexcolor}; 
                                border: solid 2px rgba(66,135,245,1);
                                border-radius: 50%;
                                height: 40px;'
                              onmouseover="document.getElementById('{hexcolor}').style.visibility='visible'"
                              onmouseout="document.getElementById('{hexcolor}').style.visibility='hidden'">
                              <div id="{hexcolor}" 
                                style='
                                  visibility: hidden;
                                  font-size: 12px; 
                                  background-color: white; 
                                  color: {hexcolor};
                                  text-align: center; 
                                  padding: 6% 6%;
                                  position: absolute; 
                                  z-index: 1;
                                  top: 120%; 
                                  left: 50%; 
                                  margin-left: -20px;
                                  '>{label}</div>
                            </div>`, 
                            iconSize: [40, 40],
                            className: 'dummy'
                        }});
                    }}
                """
            if self._index_map is not None:
                chart_colors = (
                    (0.0, 30/256, 66/256),
                    (1.0, 200/256, 44/256),
                    (0.0, 150/256, 207/256),
                    (30/256, 100/256, 0.0),
                    (172/256, 60/256, 72/256),
                )
                chart_colors += tuple(tuple((x+0.7)/2 for x in c) for c in chart_colors)
                colors = list(itertools.islice(itertools.cycle(chart_colors), len(self._cluster_labels)))
                marker_cluster = [MarkerCluster(icon_create_function = customize_marker_cluster(colors[i], label)).add_to(self._folium_map) for i, label in enumerate(self._cluster_labels)]
            else:
                marker_cluster = MarkerCluster().add_to(self._folium_map)
            clustered = True
        else:
            clustered = False
        for i, feature in enumerate(self._features.values()):
            if isinstance(feature, Circle):
                feature.draw_on(self._folium_map, self._radius_in_meters)
            elif clustered and isinstance(feature, Marker):
                if isinstance(marker_cluster, list):
                    feature.draw_on(marker_cluster[self._index_map[i]])
                else:
                    feature.draw_on(marker_cluster)
            else:
                feature.draw_on(self._folium_map)
        if self._colorbar_scale is not None: 
            scale_colors = ["#340597", "#7008a5", "#a32494", "#cf5073", "#ee7c4c", "#f69344", "#fcc22d", "#f4e82d", "#f4e82d"]
            vmin = self._colorbar_scale.pop(0)
            vmax = self._colorbar_scale.pop(-1)
            colormap = cm.LinearColormap(colors = scale_colors, index = self._colorbar_scale, caption = "*Legend above excludes outliers." if not self._include_color_scale_outliers else "", vmin = self._colorbar_scale[0], vmax = self._colorbar_scale[-1])
            self._folium_map.add_child(colormap)

    def _create_map(self):
        attrs = {'width': self._width, 'height': self._height}
        attrs.update(self._autozoom())
        attrs.update(self._attrs.copy())

        # Enforce zoom consistency
        attrs['max_zoom'] = max(attrs['zoom_start']+2, attrs['max_zoom'])
        attrs['min_zoom'] = min(attrs['zoom_start']-2, attrs['min_zoom'])
        return self._mapper(**attrs)

    def _autozoom(self):
        """Calculate zoom and location."""
        bounds = self._autobounds()
        attrs = {}

        midpoint = lambda a, b: (a + b)/2
        attrs['location'] = (
            midpoint(bounds['min_lat'], bounds['max_lat']),
            midpoint(bounds['min_lon'], bounds['max_lon'])
        )

        # remove the following with new Folium release
        # rough approximation, assuming max_zoom is 18
        import math
        try:
            lat_diff = bounds['max_lat'] - bounds['min_lat']
            lon_diff = bounds['max_lon'] - bounds['min_lon']
            area, max_area = lat_diff*lon_diff, 180*360
            if area:
                factor = 1 + max(0, 1 - self._width/1000)/2 + max(0, 1-area**0.5)/2
                zoom = math.log(area/max_area)/-factor
            else:
                zoom = self._default_zoom
            zoom = max(1, min(18, round(zoom)))
            attrs['zoom_start'] = zoom
        except ValueError as e:
            raise Exception('Check that your locations are lat-lon pairs', e)

        return attrs

    def _autobounds(self):
        """Simple calculation for bounds."""
        bounds = {}

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

        return bounds

    @property
    def features(self):
        feature_list = []
        for key, value in self._features.items():
            f = collections.OrderedDict([('id', key), ('feature', value)])
            f.update(value.properties)
            feature_list.append(f)
        return feature_list

    def format(self, **kwargs):
        """Apply formatting."""
        attrs = self._attrs.copy()
        attrs.update({'width': self._width, 'height': self._height})
        attrs.update(kwargs)
        return Map(self._features, **attrs)

    def geojson(self):
        """Render features as a FeatureCollection."""
        return {
            "type": "FeatureCollection",
            "features": [f.geojson(i) for i, f in self._features.items()]
        }

    def color(self, values, ids=(), key_on='feature.id', palette='YlOrBr', **kwargs):
        """Color map features by binning values.

        values -- a sequence of values or a table of keys and values
        ids -- an ID for each value; if none are provided, indices are used
        key_on -- attribute of each feature to match to ids
        palette -- one of the following color brewer palettes:

            'BuGn', 'BuPu', 'GnBu', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd', 'RdPu',
            'YlGn', 'YlGnBu', 'YlOrBr', and 'YlOrRd'.

        Defaults from Folium:

        threshold_scale: list, default None
            Data range for D3 threshold scale. Defaults to the following range
            of quantiles: [0, 0.5, 0.75, 0.85, 0.9], rounded to the nearest
            order-of-magnitude integer. Ex: 270 rounds to 200, 5600 to 6000.
        fill_opacity: float, default 0.6
            Area fill opacity, range 0-1.
        line_color: string, default 'black'
            GeoJSON geopath line color.
        line_weight: int, default 1
            GeoJSON geopath line weight.
        line_opacity: float, default 1
            GeoJSON geopath line opacity, range 0-1.
        legend_name: string, default None
            Title for data legend. If not passed, defaults to columns[1].
        """
        # Set values and ids to both be simple sequences by inspecting values
        id_name, value_name = 'IDs', 'values'
        if isinstance(values, collections.abc.Mapping):
            assert not ids, 'IDs and a map cannot both be used together'
            if hasattr(values, 'columns') and len(values.columns) == 2:
                table = values
                ids, values = table.columns
                id_name, value_name = table.labels
            else:
                dictionary = values
                ids, values = list(dictionary.keys()), list(dictionary.values())
        if len(ids) != len(values):
            assert len(ids) == 0
            # Use indices as IDs
            ids = list(range(len(values)))

        m = self._create_map()
        data = pandas.DataFrame({id_name: ids, value_name: values})
        attrs = {
            'geo_data': json.dumps(self.geojson()),
            'data': data,
            'columns': [id_name, value_name],
            'key_on': key_on,
            'fill_color': palette,
        }
        kwargs.update(attrs)
        folium.Choropleth(
            **kwargs,
            name='geojson'
        ).add_to(m)
        colored = self.format()
        colored._folium_map = m
        return colored

    def overlay(self, feature, color='Blue', opacity=0.6):
        """
        Overlays ``feature`` on the map. Returns a new Map.

        Args:
            ``feature``: a ``Table`` of map features, a list of map features,
                a Map, a Region, or a circle marker map table. The features will
                be overlayed on the Map with specified ``color``.

            ``color`` (``str``): Color of feature. Defaults to 'Blue'

            ``opacity`` (``float``): Opacity of overlain feature. Defaults to
                0.6.

        Returns:
            A new ``Map`` with the overlain ``feature``.
        """
        result = self.copy()
        if type(feature) == Table:
            # if table of features e.g. Table.from_records(taz_map.features)
            if 'feature' in feature:
                feature = feature['feature']

            # if marker table e.g. table with columns: latitudes,longitudes,popup,color,radius
            else:
                feature = Circle.map_table(feature)

        if type(feature) in [list, np.ndarray]:
            for f in feature:
                f._attrs['fill_color'] = color
                f._attrs['fill_opacity'] = opacity
                f.draw_on(result._folium_map)

        elif type(feature) == Map:
            for i in range(len(feature._features)):
                f = feature._features[i]
                f._attrs['fill_color'] = color
                f._attrs['fill_opacity'] = opacity
                f.draw_on(result._folium_map)
        elif type(feature) == Region:
            feature._attrs['fill_color'] = color
            feature._attrs['fill_opacity'] = opacity
            feature.draw_on(result._folium_map)
        return result

    @classmethod
    def read_geojson(cls, path_or_json_or_string_or_url):
        """Read a geoJSON string, object, file, or URL. Return a dict of features keyed by ID."""
        assert path_or_json_or_string_or_url
        data = None
        if isinstance(path_or_json_or_string_or_url, (dict, list)):
            data = path_or_json_or_string_or_url
        try:
            data = json.loads(path_or_json_or_string_or_url)
        except ValueError:
            pass
        try:
            path = path_or_json_or_string_or_url
            if path.endswith('.gz') or path.endswith('.gzip'):
                import gzip
                contents = gzip.open(path, 'r').read().decode('utf-8')
            else:
                contents = open(path, 'r').read()
            data = json.loads(contents)
        except FileNotFoundError:
            pass
        if not data:
            import urllib.request
            with urllib.request.urlopen(path_or_json_or_string_or_url) as url:
                data = json.loads(url.read().decode())
        assert data, 'MapData accepts a valid geoJSON object, geoJSON string, path to a geoJSON file, or URL'
        return cls(cls._read_geojson_features(data))

    @staticmethod
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
                value = Circle._convert_point(feature)
            elif feature_type in ['Polygon', 'MultiPolygon']:
                value = Region(feature)
            else:
                # TODO Support all http://geojson.org/geojson-spec.html#geometry-objects
                value = None
            features[key] = value
        return features


class _MapFeature(_FoliumWrapper, abc.ABC):
    """A feature displayed on a map. When displayed alone, a map is created."""

    # Default dimensions for displaying the feature in isolation
    _width = 960
    _height = 500

    def _set_folium_map(self):
        """A map containing only the feature."""
        m = Map(features=[self], width=self._width, height=self._height)
        self._folium_map = m.draw()

    #############
    # Interface #
    #############

    @property
    @abc.abstractmethod
    def lat_lons(self):
        """Sequence of lat_lons that describe a map feature (for zooming)."""

    @property
    @abc.abstractmethod
    def _folium_kwargs(self):
        """kwargs for a call to a map method."""

    @abc.abstractmethod
    def geojson(self, feature_id):
        """Return GeoJSON."""

    @abc.abstractmethod
    def draw_on(self, folium_map):
        """Add feature to Folium map object."""


class Marker(_MapFeature):
    """A marker displayed with Folium's simple_marker method.

    popup -- text that pops up when marker is clicked
    color -- The color of the marker. You can use:
    [‘red’, ‘blue’, ‘green’, ‘purple’, ‘orange’, ‘darkred’,
    ’lightred’, ‘beige’, ‘darkblue’, ‘darkgreen’, ‘cadetblue’, ‘darkpurple’, 
    ‘white’, ‘pink’, ‘lightblue’, ‘lightgreen’, ‘gray’, ‘black’, ‘lightgray’] 
    to use standard folium icons. If a hex color code is provided, 
    (color must start with '#'), a folium.plugin.BeautifyIcon will
    be used instead. 
    
    Defaults from Folium:

    marker_icon: string, default 'info-sign'
        icon from (http://getbootstrap.com/components/) you want on the
        marker
    clustered_marker: boolean, default False
        boolean of whether or not you want the marker clustered with
        other markers
    icon_angle: int, default 0
        angle of icon
    popup_width: int, default 300
        width of popup

    The icon can be further customized by by passing in attributes
    into kwargs by using the attributes listed in 
    `https://python-visualization.github.io/folium/modules.html#folium.map.Icon`.
    """

    def __init__(self, lat, lon, popup='', color='blue', **kwargs):
        assert isinstance(lat, _number)
        assert isinstance(lon, _number)
        self.lat_lon = (lat, lon)
        self._attrs = {
            'popup': popup,
            'color': color,
            **kwargs
        }
        
        # setting default icon to be empty; this is overwritten by .update()
        # on the next line if 'marker_icon' is present in kwargs
        self._attrs["marker_icon"] = "sign-blank"
        self._attrs.update(kwargs)

    @property
    def lat_lons(self):
        return [self.lat_lon]

    def copy(self):
        """Return a deep copy"""
        return type(self)(self.lat_lon[:], **self._attrs)

    @property
    def _folium_kwargs(self):
        attrs = self._attrs.copy()
        attrs['location'] = self.lat_lon
        icon_args = {k: attrs.pop(k) for k in attrs.keys() & {'color', 'marker_icon', 'clustered_marker', 'icon_angle', 'popup_width'}}
        if 'marker_icon' in icon_args:
            icon_args['icon'] = icon_args.pop('marker_icon')
        if 'color' in icon_args and icon_args['color'][0] == '#':
            # Checks if color provided is a hex code instead; if it is, uses BeautifyIcon to create markers. 
            # If statement does not check to see if color is an empty string.
            icon_args['background_color'] = icon_args['border_color'] = icon_args.pop('color')
            if icon_args['background_color'][1] == icon_args['background_color'][3] == icon_args['background_color'][5] == 'f':
                icon_args['text_color'] = 'gray'
            else:
                icon_args['text_color'] = 'white'
            icon_args['icon_shape'] = 'marker'
            if 'icon' not in icon_args:
                icon_args['icon'] = 'circle'
            attrs['icon'] = BeautifyIcon(**icon_args)
        else:
            attrs['icon'] = folium.Icon(**icon_args)
        return attrs

    def geojson(self, feature_id):
        """GeoJSON representation of the marker as a point."""
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
        """Apply formatting."""
        attrs = self._attrs.copy()
        attrs.update(kwargs)
        lat, lon = self.lat_lon
        return type(self)(lat, lon, **attrs)

    def draw_on(self, folium_map):
        folium.Marker(**self._folium_kwargs).add_to(folium_map)

    @classmethod
    def _convert_point(cls, feature):
        """Convert a GeoJSON point to a Marker."""
        lon, lat = feature['geometry']['coordinates']
        popup = feature['properties'].get('name', '')
        return cls(lat, lon)

    @classmethod
    def map(cls, latitudes, longitudes, labels=None, colors=None, areas=None, other_attrs=None, clustered_marker=False, **kwargs):
        """Return markers from columns of coordinates, labels, & colors.

        The areas column is not applicable to markers, but sets circle areas.

        Arguments: (TODO) document all options

        index_map: list of integers, default None (when not applicable)
           list of indices that maps each marker to a corresponding label at the index in cluster_labels (only applicable when multiple marker clusters are being used)

        cluster_labels: list of strings, default None (when not applicable)
            list of labels used for each cluster of markers (only applicable when multiple marker clusters are being used)

        colorbar_scale: list of floats, default None (when not applicable)
            list of cutoffs used to indicate where the bins are for each color (only applicable when colorscale gradient is being used)

        include_color_scale_outliers: boolean, default None (when not applicable)
            boolean of whether or not outliers are included in the colorscale gradient for markers (only applicable when colorscale gradient is being used)

        radius_in_meters: boolean, default False
            boolean of whether or not Circles should have their radii specified in meters, scales with map zoom

        clustered_marker: boolean, default False
            boolean of whether or not you want the marker clustered with other markers

        other_attrs: dictionary of (key) property names to (value) property values, default None
            A dictionary that list any other attributes that the class Marker/Circle should have 

        """
        assert latitudes is not None
        assert longitudes is not None
        assert len(latitudes) == len(longitudes)
        assert areas is None or hasattr(cls, '_has_radius'), "A " + cls.__name__ + " has no radius"
        inputs = [latitudes, longitudes]
        index_map = include_color_scale_outliers = cluster_labels = colorbar_scale = None
        radius_in_meters = False
        if "index_map" in kwargs:
            index_map = kwargs.pop("index_map")
        if "cluster_labels" in kwargs:
            cluster_labels = kwargs.pop("cluster_labels")
        if "colorbar_scale" in kwargs:
            colorbar_scale = kwargs.pop("colorbar_scale")
        if "include_color_scale_outliers" in kwargs:
            include_color_scale_outliers = kwargs.pop("include_color_scale_outliers")
        if "radius_in_meters" in kwargs:
            radius_in_meters = kwargs.pop("radius_in_meters")
        if labels is not None:
            assert len(labels) == len(latitudes)
            inputs.append(labels)
        else:
            inputs.append(("",) * len(latitudes))
        if colors is not None:
            assert len(colors) == len(latitudes)
            inputs.append(colors)
        if areas is not None:
            assert len(areas) == len(latitudes)
            inputs.append(np.array(areas) ** 0.5 / math.pi)
        if other_attrs is not None:
            other_attrs_processed = []
            for i in range(len(latitudes)):
                other_attrs_processed.append({})
            for prop in other_attrs:
                for i in range(len(other_attrs[prop])):
                    other_attrs_processed[i][prop] = other_attrs[prop][i]
            for dic in other_attrs_processed:
                dic.update(kwargs)
        else:
            other_attrs_processed = []
        
        if other_attrs_processed:
            ms = [cls(*args, **other_attrs_processed[row_num]) for row_num, args in enumerate(zip(*inputs))]
        else:
            ms = [cls(*args, **kwargs) for row_num, args in enumerate(zip(*inputs))]
        return Map(ms, clustered_marker=clustered_marker, index_map=index_map, cluster_labels=cluster_labels, colorbar_scale=colorbar_scale, include_color_scale_outliers=include_color_scale_outliers, radius_in_meters=radius_in_meters)

    @classmethod
    def map_table(cls, table, clustered_marker=False, include_color_scale_outliers=True, radius_in_meters=False, **kwargs):
        """Return markers from the colums of a table.
        
        The first two columns of the table must be the latitudes and longitudes
        (in that order), followed by 'labels', 'colors', 'color_scale', 'radius_scale', 'cluster_by', 'area_scale', and/or 'areas' (if applicable)
        in any order with columns explicitly stating what property they are representing.

        Args:
            ``cls``: Type of marker being drawn on the map {Marker, Circle}.
            
            ``table``: Table of data to be made into markers. The first two columns of the table must be the latitudes and longitudes (in that order), followed by 'labels', 'colors', 'cluster_by', 'color_scale', 'radius_scale', 'area_scale', and/or 'areas' (if applicable) in any order with columns explicitly stating what property they are representing. Additional columns for marker-specific attributes such as 'marker_icon' for the Marker class can be included as well.

            ``clustered_marker``: Boolean indicating if markers should be clustered with folium.plugins.MarkerCluster.

            ``include_color_scale_outliers``: Boolean indicating if outliers should be included in the color scale gradient or not. 

            ``radius_in_meters``: Boolean indicating if circle markers should be drawn to map scale or zoom scale.
        """
        lat, lon, lab, color, areas, colorbar_scale, index_map, cluster_labels, other_attrs = None, None, None, None, None, None, None, None, {}
        excluded = ["color_scale", "cluster_by", "radius_scale", "area_scale"]

        for index, col in enumerate(table.labels):
            this_col = table.column(col)
            if index == 0:
                lat = this_col
            elif index == 1:
                lon = this_col
            elif col == "labels":
                lab = this_col
            elif col == "colors":
                color = this_col
            elif col == "areas":
                areas = this_col
            elif col not in excluded:
                other_attrs[col] = this_col

        if "cluster_by" in table.labels:
            clustered_marker = True
            cluster_column = table.column("cluster_by")
            cluster_labels = list(set(cluster_column))
            table_df = table.to_df()
            table_df["indices"] = [0] * table.num_rows
            for i, label in enumerate(cluster_labels):
                table_df.loc[table_df["cluster_by"] == label, "indices"] = i
            index_map = table_df["indices"]
            del table_df
        
        if "radius_scale" in table.labels:
            radius_column = table.column("radius_scale").astype(float)
            rmin, rmax = kwargs.get("radius_min", 5), kwargs.get("radius_max", 50)
            vmin, vmax = radius_column.min(), radius_column.max()
            scale_fn = lambda v: (v - vmin) / (vmax - vmin) * (rmax - rmin) + rmin
            radii = scale_fn(radius_column)
            other_attrs["radius"] = [float(r) for r in radii]
        
        if "area_scale" in table.labels: # takes precedence over radius_scale
            area_column = table.column("area_scale").astype(float)
            amin, amax = kwargs.get("area_min", 80), kwargs.get("area_max", 8000)
            vmin, vmax = area_column.min(), area_column.max()
            scale_fn = lambda v: (v - vmin) / (vmax - vmin) * (amax - amin) + amin
            areas = scale_fn(area_column)
            radii = np.sqrt(areas / np.pi)   # convert area into radius using A = pi * r^2
            other_attrs["radius"] = [float(r) for r in radii]

        if 'color_scale' in table.labels:
            vmin = min(table.column("color_scale"))
            vmax = max(table.column("color_scale"))
            if include_color_scale_outliers:
                outlier_min_bound = vmin
                outlier_max_bound = vmax
            else:
                q1 = np.percentile(table.column("color_scale"), 25)
                q3 = np.percentile(table.column("color_scale"), 75)
                IQR = q3 - q1
                outlier_min_bound = max(vmin, q1 - 1.5 * IQR)
                outlier_max_bound = min(vmax, q3 + 1.5 * IQR)
            colorbar_scale = list(np.linspace(outlier_min_bound, outlier_max_bound, 9))
            scale_colors = ["#340597", "#7008a5", "#a32494", "#cf5073", "#ee7c4c", "#f69344", "#fcc22d", "#f4e82d", "#f4e82d"]
            def interpolate_color(colors, cutoffs, datapoint):
                for i, cutoff in enumerate(cutoffs):
                    if cutoff >= datapoint:
                        return colors[i - 1] if i > 0 else colors[0]
                return colors[-1]
            color = [""] * table.num_rows
            for i, datapoint in enumerate(table.column('color_scale')): 
                color[i] = interpolate_color(scale_colors, colorbar_scale, datapoint)
            colorbar_scale = [vmin] + colorbar_scale + [vmax]
        if not other_attrs:
            other_attrs = None
        return cls.map(
            latitudes=lat, longitudes=lon, labels=lab, colors=color, areas=areas, 
            colorbar_scale=colorbar_scale, other_attrs=other_attrs, clustered_marker=clustered_marker, 
            index_map=index_map, include_color_scale_outliers=include_color_scale_outliers, 
            radius_in_meters=radius_in_meters, cluster_labels=cluster_labels, **kwargs
        )

class Circle(Marker):
    """A marker displayed with either Folium's circle_marker or circle methods.

    The ``circle_marker`` method draws circles that stay the same size regardless of map zoom, 
    whereas the circle method draws circles that have a fixed radius in meters. To toggle 
    between them, use the ``radius_in_meters`` flag in the draw_on function. 

    popup -- text that pops up when marker is clicked
    color -- fill color
    radius -- pixel radius of the circle

    Defaults from Folium:

    fill_opacity: float, default 0.6
        Circle fill opacity

    More options can be passed into kwargs by following the attributes
    listed in `https://leafletjs.com/reference-1.4.0.html#circlemarker` or 
    `https://leafletjs.com/reference-1.4.0.html#circle`.

    For example, to draw three circles with circle_marker:

    ..code-block:: python

        t = Table().with_columns([
                'lat', [37.8, 38, 37.9],
                'lon', [-122, -122.1, -121.9],
                'label', ['one', 'two', 'three'],
                'color', ['red', 'green', 'blue'],
                'radius', [3000, 4000, 5000],
            ])
        Circle.map_table(t)

    To draw three circles with the circle methods, replace the last line with:

    ..code-block:: python
    
        Circle.map_table(t, radius_in_meters=True)
    """

    _has_radius = True

    def __init__(self, lat, lon, popup='', color='blue', radius=10, **kwargs):
        super().__init__(lat, lon, popup, color, radius=float(radius), **kwargs)

    @property
    def _folium_kwargs(self):
        attrs = self._attrs.copy()
        attrs['location'] = self.lat_lon
        if 'color' in attrs:
            attrs['fill_color'] = attrs.pop('color')
        if 'line_color' in attrs:
            attrs['color'] = attrs.pop('line_color')
        return attrs

    def draw_on(self, folium_map, radius_in_meters=False):
        if radius_in_meters:
            folium.Circle(**self._folium_kwargs).add_to(folium_map)
        else:
            folium.CircleMarker(**self._folium_kwargs).add_to(folium_map)


class Region(_MapFeature):
    """A GeoJSON feature displayed with Folium's geo_json method."""

    def __init__(self, geojson, **kwargs):
        assert 'type' in geojson
        assert geojson['type'] == 'Feature'
        assert 'geometry' in geojson
        assert 'type' in geojson['geometry']
        assert geojson['geometry']['type'] in ['Polygon', 'MultiPolygon']
        self._geojson = geojson
        self._attrs = kwargs

    @property
    def lat_lons(self):
        """A flat list of (lat, lon) pairs."""
        return _lat_lons_from_geojson(self._geojson['geometry']['coordinates'])

    @property
    def type(self):
        """The GEOJSON type of the regions: Polygon or MultiPolygon."""
        return self._geojson['geometry']['type']

    @property
    def polygons(self):
        """Return a list of polygons describing the region.

        - Each polygon is a list of linear rings, where the first describes the
          exterior and the rest describe interior holes.
        - Each linear ring is a list of positions where the last is a repeat of
          the first.
        - Each position is a (lat, lon) pair.
        """
        if self.type == 'Polygon':
            polygons = [self._geojson['geometry']['coordinates']]
        elif self.type == 'MultiPolygon':
            polygons = self._geojson['geometry']['coordinates']
        return [   [   [_lat_lons_from_geojson(s) for
                        s in ring  ]              for
                    ring in polygon]              for
                polygon in polygons]

    @property
    def properties(self):
        return self._geojson.get('properties', {})

    def copy(self):
        """Return a deep copy"""
        return type(self)(self._geojson.copy(), **self._attrs)

    @property
    def _folium_kwargs(self):
        attrs = self._attrs.copy()
        attrs['data'] = json.dumps(self._geojson)
        return attrs

    def geojson(self, feature_id):
        """Return GeoJSON with ID substituted."""
        if self._geojson.get('id', feature_id) == feature_id:
            return self._geojson
        else:
            geo = self._geojson.copy()
            geo['id'] = feature_id
            return geo

    def format(self, **kwargs):
        """Apply formatting."""
        attrs = self._attrs.copy()
        attrs.update(kwargs)
        return Region(self._geojson, **attrs)

    def draw_on(self, folium_map):
        attrs = self._folium_kwargs
        data = attrs.pop('data')
        folium.GeoJson(
            data=data,
            style_function=lambda x: attrs,
            name='geojson'
        ).add_to(folium_map)


def _lat_lons_from_geojson(s):
    """Return a latitude-longitude pairs from nested GeoJSON coordinates.

    GeoJSON coordinates are always stored in (longitude, latitude) order.
    """
    if len(s) >= 2 and isinstance(s[0], _number) and isinstance(s[0], _number):
        lat, lon = s[1], s[0]
        return [(lat, lon)]
    else:
        return [lat_lon for sub in s for lat_lon in _lat_lons_from_geojson(sub)]


def get_coordinates(table, replace_columns=False, remove_nans=False):
    """
    Adds latitude and longitude coordinates to table based on other location identifiers. Must be in the United States.

    Takes table with columns "zip code" or "city" and/or "county" and "state" in column names and 
    adds the columns "lat" and "lon". If a county is not found inside the dataset,
    that row's latitude and longitude coordinates are replaced with np.nans. The 'replace_columns' flag
    indicates if the "city", "county", "state", and "zip code" columns should be removed afterwards.
    The 'remove_nans' flag indicates if rows with nan latitudes and longitudes should be removed. Robust to 
    capitalization.

    Dataset was acquired on July 2, 2020 from https://docs.gaslamp.media/download-zip-code-latitude-longitude-city-state-county-csv. 
    Found in geocode_datasets/geocode_states.csv. Modified column names and made city/county columns all in lowercase. 

    Args:
        table: A table with counties that need to mapped to coordinates
        replace_columns: A boolean that indicates if "county", "city", "state", and "zip code" columns should be removed 
        remove_nans: A boolean that indicates if columns with invalid longitudes and latitudes should be removed
        
    Returns:
        Table with latitude and longitude coordinates 
    """
    assert "zip code" in table.labels or (("city" in table.labels or "county" in table.labels) and "state" in table.labels)
    ref = pandas.read_csv(pkg_resources.resource_filename(__name__, "geodata/geocode_states.csv"))
    table_df = table.to_df()
    table_df["lat"] = [np.nan] * table.num_rows
    table_df["lon"] = [np.nan] * table.num_rows
    unassigned = set(range(table.num_rows)) # Indices where latitudes and longitudes have not been assigned yet
    while len(unassigned) > 0:
        index = unassigned.pop()
        df_row = table_df.iloc[index]
        if "zip code" in table_df.columns:
            select = table_df["zip code"] == df_row["zip code"]
            unassigned -= set(table_df.index[select])
            try:
                compared = ref["zip"] == int(df_row["zip code"])
                table_df.loc[select, "lat"] = ref.loc[compared, "lat"].tolist()[0]
                table_df.loc[select, "lon"] = ref.loc[compared, "lon"].tolist()[0]
            except (IndexError, ValueError):
                pass
        else:
            state_select = table_df["state"] == df_row["state"]
            county_select = table_df["county"] == df_row["county"] if "county" in table_df.columns else np.array([True] * table.num_rows)
            city_select = table_df["city"] == df_row["city"] if "city" in table_df.columns else np.array([True] * table.num_rows)
            select = state_select & county_select & city_select
            unassigned -= set(table_df.index[select])
            try: 
                lowered_county = None if "county" not in table_df.columns else df_row["county"].lower()
                lowered_city = None if "city" not in table_df.columns else df_row["city"].lower()
                if "county" in table_df.columns and "city" not in table_df.columns: 
                    compared = (ref["state"] == df_row["state"]) & (ref["county"] == lowered_county)
                    table_df.loc[select, "lat"] = ref.loc[compared, "lat"].tolist()[0]
                    table_df.loc[select, "lon"] = ref.loc[compared, "lon"].tolist()[0]
                elif "county" not in table_df.columns and "city" in table_df.columns:
                    compared = (ref["state"] == df_row["state"]) & (ref["city"] == lowered_city)
                    table_df.loc[select, "lat"] = ref.loc[compared, "lat"].tolist()[0]
                    table_df.loc[select, "lon"] = ref.loc[compared, "lon"].tolist()[0]
                else:
                    compared = (ref["state"] == df_row["state"]) & (ref["county"] == lowered_county) & (ref["city"] == lowered_city)
                    table_df.loc[select, "lat"] = ref.loc[compared, "lat"].tolist()[0]
                    table_df.loc[select, "lon"] = ref.loc[compared, "lon"].tolist()[0]
            except IndexError:
                pass
    if replace_columns:
        for label in ["county", "city", "zip code", "state"]:
            try:
                table_df.drop(label, axis = 1, inplace = True)
            except KeyError:
                pass
    if remove_nans: 
        table_df.dropna(subset=["lat", "lon"], inplace = True)
    return Table.from_df(table_df)
