import doctest
import json
import pytest
import unittest
import math
import numpy as np
from collections import OrderedDict

import datascience as ds
from datascience import maps


@pytest.fixture(scope='function')
def states():
    """Read a map of US states."""
    return ds.Map.read_geojson('tests/us-states.json')


############
# Doctests #
############


def test_doctests():
    results = doctest.testmod(maps, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert results.failed == 0


############
# Overview #
############


def test_draw_map(states):
    """ Tests that draw_map returns HTML """
    states.show()


def test_setup_map():
    """ Tests that passing kwargs doesn't error. """
    kwargs1 = {
        'tiles': 'Stamen Toner',
        'zoom_start': 17,
        'width': 960,
        'height': 500,
        'features': [],
        'attr':'Map tiles by Stamen Design, CC BY 3.0 - Map data © OpenStreetMap'
    }
    """ Tests features as NumPy array. """
    kwargs2 = {
        'tiles': 'Stamen Toner',
        'zoom_start': 17,
        'width': 960,
        'height': 500,
        'features': np.array([
            ds.Marker(51.514, -0.132), 
            ds.Marker(51.514, -0.139), 
            ds.Marker(51.519, -0.132)
        ]),
        'attr':'Map tiles by Stamen Design, CC BY 3.0 - Map data © OpenStreetMap'
    }
    ds.Map(**kwargs1).show()
    ds.Map(**kwargs2).show()


def test_map_marker_and_region(states):
    """ Tests that a Map can contain a Marker and/or Region. """
    marker = ds.Marker(51.514, -0.132)
    ds.Map(marker).show()
    ds.Map([marker]).show()
    region = states['CA']
    ds.Map(region).show()
    ds.Map([region]).show()
    ds.Map([marker, region]).show()

def test_map_property_features(states):
    feature_list = states.features
    assert isinstance(feature_list, list)
    assert isinstance(feature_list[0], OrderedDict)
    assert feature_list[0]['id'] == 'AL'
    assert feature_list[0]['name'] == 'Alabama'
    assert isinstance(feature_list[0]['feature'], maps.Region)
    tt = [list(i.keys()) == ['id', 'feature', 'name'] for i in feature_list]
    assert all(tt)

def test_map_copy(states):
    """Tests that copy returns a copy of the current map"""
    
    map1 = states
    map2 = map1.copy()

    # Compare geojsons of the two map objects
    assert map1.geojson() == map2.geojson() 
    # Assert that map1 and map2 not the same object
    # and copy is returning a true copy
    assert map1 is not map2

def test_map_overlay_undefined_feature():
    marker1 = ds.Marker(51.514, -0.132)
    marker2 = ds.Marker(52.514, -0.132)
    marker1_map = ds.Map(marker1)
    unchanged_map = marker1_map.overlay(marker2)
    assert len(unchanged_map._features), 1
    assert len(unchanged_map._folium_map._children.keys()), 1
    marker2_map = ds.Map(marker2)
    changed_map = marker1_map.overlay(marker2_map)
    assert len(changed_map._features), 1
    assert len(unchanged_map._folium_map._children.keys()), 2



#############
# ds.Marker #
#############


def test_marker_html():
    """ Tests that a Marker can be rendered. """
    ds.Marker(51.514, -0.132).show()


def test_marker_map():
    """ Tests that Marker.map generates a map """
    lats = [51, 52, 53]
    lons = [-1, -2, -3]
    labels = ['A', 'B', 'C']
    colors = ['blue', 'red', 'green']
    ds.Marker.map(lats, lons).show()
    ds.Marker.map(lats, lons, labels).show()
    ds.Marker.map(lats, lons, labels, colors).show()
    ds.Marker.map(lats, lons, colors=colors).show()


def test_marker_map_table():
    """ Tests that Marker.map_table generates a map """
    lats = [51, 52, 53]
    lons = [-1, -2, -3]
    labels = ['A', 'B', 'C']
    t = ds.Table().with_columns('A', lats, 'B', lons, 'C', labels)
    ds.Marker.map_table(t).show()
    colors = ['red', 'green', 'yellow']
    t['colors'] = colors
    markers = ds.Marker.map_table(t)

    assert markers[0]._attrs['color'], 'red'
    assert markers[1]._attrs['color'], 'green'
    assert markers[2]._attrs['color'], 'yellow'

    assert markers[0].lat_lon[0], 51
    assert markers[1].lat_lon[0], 52
    assert markers[2].lat_lon[0], 53

    assert markers[0].lat_lon[1], -1
    assert markers[1].lat_lon[1], -2
    assert markers[2].lat_lon[1], -3

def test_marker_copy():
    lat, lon = 51, 52
    a = ds.Marker(lat, lon)
    b = a.copy()
    b_lat_lon = b.lat_lon
    assert lat == b_lat_lon[0]
    assert lon == b_lat_lon[1]

def test_background_color_condition_white():
    # Test the condition when the background color is white (all 'f' in the hex code)
    marker = ds.Marker(0, 0, color='#ffffff')
    assert marker._folium_kwargs['icon'].options['textColor'], 'gray'

def test_background_color_condition_not_white():
    # Test the condition when the background color is not white
    marker = ds.Marker(0, 0, color='#ff0000')
    assert marker._folium_kwargs['icon'].options['textColor'], 'white'

def test_icon_args_icon_not_present():
    # Test when 'icon' key is not present in icon_args
    marker = ds.Marker(0, 0, color='blue', marker_icon='info-sign')
    assert marker._folium_kwargs['icon'].options['icon'], 'circle'

def test_icon_args_icon_present():
    # Test when 'icon' key is already present in icon_args
    marker = ds.Marker(0, 0, color='blue', marker_icon='info-sign', icon='custom-icon')
    assert marker._folium_kwargs['icon'].options['icon'], 'info-sign'

def test_user_tampered_marker_icon_attributes():
    marker = ds.Marker(0, 0, color='#ff0000')
    del marker._attrs["marker_icon"]
    assert marker._folium_kwargs['icon'].options['icon'], 'circle'

def test_geojson():
    # Create a Marker instance with known values
    marker = ds.Marker(lat=40.7128, lon=-74.0060, popup="New York City")
    # Define a feature_id for testing
    feature_id = 1
    # Call the geojson method to get the GeoJSON representation
    geojson = marker.geojson(feature_id)
    # Define the expected GeoJSON representation
    expected_geojson = {
        'type': 'Feature',
        'id': feature_id,
        'geometry': {
            'type': 'Point',
            'coordinates': (-74.0060, 40.7128),
        },
    }
    # Compare the actual and expected GeoJSON representations
    assert geojson, expected_geojson

def test_convert_point():
    feature = {
        'geometry': {
            'coordinates': [12.34, 56.78],
        },
        'properties': {
            'name': 'Test Location',
        }
    }
    converted_marker = ds.Marker._convert_point(feature)
    assert converted_marker.lat_lon, (56.78, 12.34)
    assert converted_marker._attrs['popup'], 'Test Location'

def test_convert_point_no_name():
    feature = {
        'geometry': {
            'coordinates': [98.76, 54.32],
        },
        'properties': {}
    }
    converted_marker = ds.Marker._convert_point(feature)
    assert converted_marker.lat_lon, (54.32, 98.76)
    assert not converted_marker._attrs['popup']


##########
# Circle #
##########

def test_line_color_handling():
    # Create a Circle instance with line_color attribute
    circle = ds.Circle(37.8, -122, line_color='red')
    # Call the _folium_kwargs method to get the attributes
    attrs = circle._folium_kwargs
    # Check that 'line_color' attribute has been transferred to 'color'
    assert attrs['color'], 'red'

def test_circle_map_table():
    lat_init, lon_init, area_init, color_scale_init = 51, -8, 10, 10
    lats, lons, areas, color_scales = [], [], [], []
    for i in range(8):
        lats.append(lat_init+i)
        lons.append(lon_init+i)
        areas.append((area_init + 10*i)**2*math.pi)
        color_scales.append(color_scale_init + 10*i)
    color_scales[-1] = 1000000
    labels = ['A', 'B', 'C']
    t = ds.Table().with_columns('A', lats, 'B', lons, 'areas', areas, 'color_scale', color_scales)
    markers = ds.Circle.map_table(t, include_color_scale_outliers=False)

    for i in range(8):
        assert markers[i]._attrs['radius'], 10 + 10*i

    # Call the map_table method and check if percentiles and outliers are calculated correctly
    assert markers._attrs['colorbar_scale'], [10.0, 23.125, 36.25, 49.375, 62.5, 75.625, 88.75, 101.875, 115.0]

    assert [markers[i]._attrs['color'] for i in range(8)], ['#340597', '#340597', '#7008a5', '#a32494', '#cf5073', '#cf5073', '#ee7c4c', '#f4e82d']


def test_circle_html():
    """ Tests that a Circle can be rendered. """
    ds.Circle(51.514, -0.132).show()


def test_circle_map():
    """ Tests that Circle.map generates a map """
    lats = [51, 52, 53]
    lons = [-1, -2, -3]
    labels = ['A', 'B', 'C']
    ds.Circle.map(lats, lons).show()
    ds.Circle.map(lats, lons, labels).show()

def test_user_tampered_circle_color_attributes():
    circle = ds.Circle(51.514, -0.132)
    del circle._attrs["color"]
    assert "color" not in circle._folium_kwargs


##########
# Bounds #
##########


def test_bounds():
    """ Tests that generated bounds are correct """
    points = [ds.Marker(0, 0), ds.Marker(-89.9, 180), ds.Marker(90, -180)]
    bounds = ds.Map(points)._autobounds()
    assert bounds['max_lat'] == 90
    assert bounds['min_lat'] == -89.9
    assert bounds['max_lon'] == 180
    assert bounds['min_lon'] == -180


def test_bounds_limits():
    """ Tests that too-large lats and lons are truncated to real bounds. """
    points = [ds.Marker(0, 0), ds.Marker(-190, 280), ds.Marker(190, -280)]
    bounds = ds.Map(points)._autobounds()
    assert bounds['max_lat'] == 90
    assert bounds['min_lat'] == -90
    assert bounds['max_lon'] == 180
    assert bounds['min_lon'] == -180


#########
# Color #
#########


def test_color_table(states):
    """ Tests that color can take a Table. """
    data = ds.Table.read_table('tests/us-unemployment.csv')
    states.color(data).show()


def test_color_dict(states):
    """ Tests that color can take a dict. """
    data = ds.Table.read_table('tests/us-unemployment.csv')
    states.color(dict(zip(*data.columns))).show()


def test_color_values_and_ids(states):
    """ Tests that color can take values and ids. """
    data = ds.Table.read_table('tests/us-unemployment.csv')
    states.color(data['Unemployment'], data['State']).show()

def test_color_with_ids(states):
    # Case number of values and ids are different
    states.color([1, 2, 3, 4, 5], []).show()

###########
# GeoJSON #
###########

def test_read_geojson_with_dict():
    data = {'type': 'FeatureCollection', 'features': []}
    map_data = ds.Map.read_geojson(data)
    assert isinstance(map_data, ds.Map)

def test_read_geojson_with_gz_file():
    map_data = ds.Map.read_geojson('tests/us-states-zipped.json.gz')
    assert isinstance(map_data, ds.Map)

def test_read_geojson_features_with_valid_data():
    data = {
        'type': 'FeatureCollection',
        'features': [
            {
                'id': '1',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [125.6, 10.1]
                },
                'properties': {
                    'name': 'Dinagat Islands'
                }
            }
        ]
    }
    features = ds.Map._read_geojson_features(data)
    assert '1' in features
    assert isinstance(features['1'], ds.Circle)

def test_read_geojson_features_with_nested_feature_collection():
    data = {
        'type': 'FeatureCollection',
        'features': [
            {
                'id': '1',
                'geometry': {
                    'type': 'FeatureCollection',
                    'features': [
                        {
                            'id': '1.1',
                            'geometry': {
                                'type': 'Point',
                                'coordinates': [125.6, 10.1]
                            },
                            'properties': {
                                'name': 'Dinagat Islands'
                            }
                        }
                    ]
                },
                'properties': {
                    'name': 'Philippines'
                }
            }
        ]
    }
    features = ds.Map._read_geojson_features(data)
    assert '1' in features
    assert '1.1' in features
    assert isinstance(features['1.1'], ds.Circle)

def test_read_geojson_features_with_invalid_geometry_type():
    data = {
        'type': 'FeatureCollection',
        'features': [
            {
                'id': '1',
                'geometry': {
                    'type': 'InvalidType',
                    'coordinates': [125.6, 10.1]
                },
                'properties': {
                    'name': 'Dinagat Islands'
                }
            }
        ]
    }
    features = ds.Map._read_geojson_features(data)
    assert '1' in features
    assert features['1'] is None

##########
# Region #
##########

def test_region_html(states):
    states['CA'].show()


def test_geojson(states):
    """ Tests that geojson returns the original US States data """
    data = json.load(open('tests/us-states.json', 'r'))
    geo = states.geojson()
    assert data == geo, '{}\n{}'.format(data, geo)

def test_polygon_type():
    # Test if polygons property returns the correct structure for 'Polygon' type
    geojson_polygon = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
        }
    }
    geojson_multi_polygon = {
        "type": "Feature",
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                [[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]]
            ]
        }
    }
    region_polygon = ds.Region(geojson_polygon)
    region_multi_polygon = ds.Region(geojson_multi_polygon)
    assert region_polygon.type, "Polygon"
    assert region_multi_polygon.type, "MultiPolygon"

    # Test if polygons property returns the correct structure for 'Polygon' type
    polygons = region_polygon.polygons
    assert len(polygons), 1
    assert len(polygons[0]), 1  # One polygon
    assert len(polygons[0][0]), 5  # Five points (closed ring)

    # Test if polygons property returns the correct structure for 'MultiPolygon' type
    polygons = region_multi_polygon.polygons
    assert len(polygons), 2  # Two polygons
    for polygon in polygons:
        assert len(polygon), 1  # Each with one ring
        assert len(polygon[0]), 5  # Five points (closed ring)

def test_copy_method():
    # Set up sample GeoJSON object and attributes
    geojson = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
        },
        "properties": {"name": "Test Region"}
    }
    attrs = {"color": "red"}
    # Create a Region object
    region = ds.Region(geojson, **attrs)
    # Use the copy method to create a deep copy
    copied_region = region.copy()
    # Check if the copied region has the same attributes as the original
    assert copied_region._geojson, geojson.copy()
    assert copied_region._attrs, attrs

def test_geojson_with_id():
    # Create a sample Region object with a GeoJSON object
    geojson_data = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
        },
        "properties": {
            "name": "Test Region"
        },
        "id": "test_id"
    }
    region = ds.Region(geojson_data)
    # Call the geojson method with a new feature_id
    updated_geojson = region.geojson("new_id")
    # Check that the new ID is correctly set in the returned geojson
    assert updated_geojson["id"], "new_id"
    # Check that the other properties of the GeoJSON are retained
    assert updated_geojson["type"], geojson_data["type"]
    assert updated_geojson["geometry"], geojson_data["geometry"]
    assert updated_geojson["properties"], geojson_data["properties"]

###################
# get_coordinates #
###################

def test_remove_nonexistent_county_column():
    # Create a table without the "county" column
    data = {'city': ['City1', 'City2'], 'state': ['State1', 'State2']}
    table = ds.Table().with_columns(data)
    # Call get_coordinates with remove_columns=True
    result = maps.get_coordinates(table, replace_columns=True)
    # Ensure that the "county" column is removed
    assert 'county' not in result.labels

def test_remove_nonexistent_city_column():
    # Create a table without the "city" column
    data = {'county': ['County1', 'County2'], 'state': ['State1', 'State2']}
    table = ds.Table().with_columns(data)
    # Call get_coordinates with remove_columns=True
    result = maps.get_coordinates(table, replace_columns=True)
    # Ensure that the "city" column is removed
    assert 'city' not in result.labels

def test_remove_nonexistent_zip_code_column():
    # Create a table without the "zip code" column
    data = {'county': ['County1', 'County2'], 'state': ['State1', 'State2']}
    table = ds.Table().with_columns(data)
    # Call get_coordinates with remove_columns=True
    result = maps.get_coordinates(table, replace_columns=True)
    # Ensure that the "zip code" column is removed
    assert 'zip code' not in result.labels
