"""
This file has extra tests to make sure we cover more of the code.
It's mostly for checking edge cases and making sure everything works as expected.
"""
import numpy as np
import datascience as ds
from datascience import maps
from datascience.tables import Table

def test_marker_repr_html():
    """Test that Marker has proper _repr_html_ implementation."""
    # This just checks if the HTML representation is a string and looks reasonable.
    marker = ds.Marker(51.514, -0.132)
    html = marker._repr_html_()
    assert isinstance(html, str)
    assert 'marker' in html.lower() or 'folium' in html.lower()

def test_circle_repr_html():
    """Test that Circle has proper _repr_html_ implementation."""
    circle = ds.Circle(51.514, -0.132)
    html = circle._repr_html_()
    assert isinstance(html, str)
    assert 'circle' in html.lower() or 'folium' in html.lower()

def test_region_repr_html():
    """Test that Region has proper _repr_html_ implementation."""
    states = ds.Map.read_geojson('tests/us-states.json')
    region = states['CA']
    html = region._repr_html_()
    assert isinstance(html, str)

def test_map_repr_html():
    """Test that Map has proper _repr_html_ implementation."""
    states = ds.Map.read_geojson('tests/us-states.json')
    html = states._repr_html_()
    assert isinstance(html, str)
    assert 'folium' in html.lower()

def test_inline_map():
    """Test the _inline_map static method."""
    # This is an internal method, but we test it for coverage.
    states = ds.Map.read_geojson('tests/us-states.json')
    html = ds.maps._FoliumWrapper._inline_map(states, 800, 600)
    assert isinstance(html, str)
    assert 'iframe' in html.lower() or '<div' in html.lower() or '<html' in html.lower()

def test_map_with_numpy_array():
    """Test Map initialization with numpy array of features."""
    # Can we create a Map from a numpy array?
    features_array = np.array([
        ds.Marker(51.514, -0.132),
        ds.Marker(51.514, -0.139),
        ds.Marker(51.519, -0.132)
    ])
    map_from_array = ds.Map(features=features_array)
    assert len(map_from_array) == 3

def test_map_with_features_dict():
    """Test Map initialization with dictionary of features."""
    # Can we create a Map from a dictionary of features?
    features_dict = {
        'loc1': ds.Marker(51.514, -0.132),
        'loc2': ds.Circle(51.514, -0.139),
    }
    map_from_dict = ds.Map(features=features_dict, tiles='CartoDB dark_matter')
    assert len(map_from_dict) == 2

def test_single_feature():
    """Test Map with single feature (non-dict case)."""
    # What happens if we just pass a single feature to the Map?
    marker = ds.Marker(51.514, -0.132)
    map_from_single = ds.Map(marker)
    assert len(map_from_single) == 1

def test_table_to_markers():
    """Test Marker creation from table."""
    # Test making a map of markers from a table of data.
    t = Table().with_columns(
        'lat', [51, 52, 53],
        'lon', [-1, -2, -3],
        'labels', ['Loc1', 'Loc2', 'Loc3']
    )
    markers_map = ds.Marker.map_table(t)
    assert len(markers_map) == 3

def test_table_to_circles():
    """Test Circle creation from table."""
    # Test making a map of circles from a table.
    t = Table().with_columns(
        'lat', [51, 52, 53],
        'lon', [-1, -2, -3],
        'areas', [10, 20, 30]
    )
    circles_map = ds.Circle.map_table(t, include_color_scale_outliers=False)
    assert isinstance(circles_map, ds.maps.Map)

def test_marker_copy_deep():
    """Test deep copy of marker preserves attributes."""
    # Make sure that when we copy a marker, the original isn't changed.
    original_marker = ds.Marker(51.514, -0.132, popup='Test marker', color='blue')
    copied_marker = original_marker.copy()
    copied_marker.lat_lon = (50, 0)
    assert original_marker.lat_lon == (51.514, -0.132)

def test_circle_copy_deep():
    """Test deep copy of circle preserves attributes."""
    # Make sure that when we copy a circle, the original isn't changed.
    original_circle = ds.Circle(51.514, -0.132, color='red', popup='Test circle')
    copied_circle = original_circle.copy()
    copied_circle.lat_lon = (50, 0)
    assert original_circle.lat_lon == (51.514, -0.132)

def test_map_color_dict():
    """Test Map.color method with dictionary input."""
    # Test coloring a map of states based on unemployment data.
    states = ds.Map.read_geojson('tests/us-states.json')
    data = ds.Table.read_table('tests/us-unemployment.csv')
    colors = {row.item('State'): row.item('Unemployment') for row in data.rows}
    mapped_states = states.color(colors)
    assert len(mapped_states) == len(states)

def test_markers_with_popup():
    """Test markers with popup text."""
    # Check if popups are assigned correctly.
    markers_map = ds.Marker.map([51.514, 51.515], [-0.132, -0.133], ['London', 'Paris'])
    features = list(markers_map.values())
    assert features[0]._attrs['popup'] == 'London'
    assert features[1]._attrs['popup'] == 'Paris'

def test_markers_with_custom_colors():
    """Test markers with custom colors."""
    # Check if custom colors are assigned correctly.
    markers_map = ds.Marker.map([51.514, 51.515], [-0.132, -0.133], colors=['red', 'blue'])
    features = list(markers_map.values())
    assert features[0]._attrs['color'] == 'red'
    assert features[1]._attrs['color'] == 'blue'

def test_convert_nested_feature_collection():
    """Test feature collection conversion with nested GeoJSON."""
    # Test a more complex GeoJSON structure.
    data = {
        'type': 'FeatureCollection',
        'features': [{'type': 'Feature', 'id': '1', 'geometry': {'type': 'Polygon', 'coordinates': [[[0,0], [1,0], [1,1], [0,1], [0,0]]]}, 'properties': {'name': 'Test'}}]
    }
    features = ds.Map._read_geojson_features(data)
    assert '1' in features
    assert isinstance(features['1'], ds.Region)

def test_convert_point_with_popup():
    """Test point conversion with popup in properties."""
    # Test converting a single GeoJSON point to a marker.
    feature = {
        'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [12.34, 56.78]},
        'properties': {'name': 'Test Location', 'popup': 'Custom Popup'}
    }
    converted_marker = ds.Marker._convert_point(feature)
    assert converted_marker.lat_lon == (56.78, 12.34)
    # The code uses the 'name' for the popup, not the 'popup' property itself.
    assert converted_marker._attrs['popup'] == 'Test Location'

def test_convert_point_without_name():
    """Test point conversion without named property."""
    # What if the point has no 'name' property for the popup?
    feature = {
        'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [12.34, 56.78]},
        'properties': {}
    }
    converted_marker = ds.Marker._convert_point(feature)
    assert converted_marker.lat_lon == (56.78, 12.34)
    assert converted_marker._attrs['popup'] == ''

def test_invalid_geometry_type():
    """Test conversion of invalid geometry type returns None."""
    # It should handle weird geometry types gracefully without crashing.
    data = {
        'type': 'FeatureCollection',
        'features': [{'type': 'Feature', 'id': '1', 'geometry': {'type': 'InvalidType', 'coordinates': [12.34, 56.78]}, 'properties': {'name': 'Test'}}]
    }
    features = ds.Map._read_geojson_features(data)
    assert features['1'] is None

def test_region_polygon_structure():
    """Test that Region preserves correct polygon structure."""
    # Check the structure of a polygon region is what we expect.
    polygon_data = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[0,0], [1,0], [1,1], [0,1], [0,0]]]}}
    region = ds.Region(polygon_data)
    polygons = region.polygons
    first_point = polygons[0][0][0][0]
    assert len(first_point) == 2

def test_region_multipolygon_structure():
    """Test that Region MultiPolygon has correct structure."""
    # Check the structure of a multi-polygon region.
    multi_polygon_data = {
        "type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": [[[[0,0], [1,0], [1,1], [0,1], [0,0]]], [[[2,2], [3,2], [3,3], [2,3], [2,2]]]]},
        "properties": {'name': 'Test'}
    }
    region = ds.Region(multi_polygon_data)
    polygons = region.polygons
    first_point_poly1 = polygons[0][0][0][0]
    first_point_poly2 = polygons[1][0][0][0]
    assert first_point_poly1 == (0, 0)
    assert first_point_poly2 == (2, 2)

def test_map_bounds_normal():
    """Test normal map bounds calculation."""
    # Make sure the automatic map bounds are calculated correctly.
    markers = [ds.Marker(0, 0), ds.Marker(51, -122), ds.Marker(37, -122)]
    bounds = ds.Map(markers)._autobounds()
    assert bounds['min_lat'] == 0 and bounds['max_lat'] == 51
    assert bounds['min_lon'] == -122 and bounds['max_lon'] == 0

def test_map_bounds_at_limits():
    """Test map bounds at extreme lat/lon values."""
    # Test with coordinates at the edges of the world map.
    markers = [ds.Marker(-90, -180), ds.Marker(90, 180)]
    bounds = ds.Map(markers)._autobounds()
    assert bounds['min_lat'] == -90 and bounds['max_lat'] == 90
    assert bounds['min_lon'] == -180 and bounds['max_lon'] == 180

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, '-v', '--cov=datascience', '--cov-report=term-missing'])
