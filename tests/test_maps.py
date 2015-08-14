from datascience import *
from unittest.mock import MagicMock, patch
import pytest
import copy
from IPython.display import HTML


@pytest.fixture(scope='function')
def states():
    """Read a map of US states."""
    return Map.read_geojson('tests/us-states.json')


############
# Overview #
############


def test_draw_map(states):
    """ Tests that draw_map returns HTML """
    states.show()


def test_setup_map():
    """ Tests that passing kwargs doesn't error. """
    kwargs = {
        'tiles': 'Stamen Toner',
        'zoom_start': 17,
        'width': 960,
        'height': 500,
        'features': [],
    }
    Map(**kwargs).show()


def test_map_marker_and_region(states):
    """ Tests that a Map can contain a Marker and/or Region. """
    marker = Marker(51.514, -0.132)
    Map(marker).show()
    Map([marker]).show()
    region = states['CA']
    Map(region).show()
    Map([region]).show()
    Map([marker, region]).show()


##########
# Marker #
##########


def test_marker_html():
    """ Tests that a Marker can be rendered. """
    Marker(51.514, -0.132).show()


def test_marker_map():
    """ Tests that Marker.map generates a map """
    lats = [51, 52, 53]
    lons = [-1, -2, -3]
    labels = ['A', 'B', 'C']
    colors = ['blue', 'red', 'green']
    Marker.map(lats, lons).show()
    Marker.map(lats, lons, labels).show()
    Marker.map(lats, lons, labels, colors).show()
    Marker.map(lats, lons, colors=colors).show()


def test_marker_map_table():
    """ Tests that Marker.map_table generates a map """
    lats = [51, 52, 53]
    lons = [-1, -2, -3]
    labels = ['A', 'B', 'C']
    t = Table([lats, lons, labels], ['A', 'B', 'C'])
    Marker.map_table(t).show()
    colors = ['red', 'green', 'yellow']
    t['colors'] = colors
    Marker.map_table(t).show()


def test_circle_html():
    """ Tests that a Circle can be rendered. """
    Circle(51.514, -0.132).show()


def test_circle_map():
    """ Tests that Circle.map generates a map """
    lats = [51, 52, 53]
    lons = [-1, -2, -3]
    labels = ['A', 'B', 'C']
    Circle.map(lats, lons).show()
    Circle.map(lats, lons, labels).show()


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


##########
# Bounds #
##########


def test_bounds():
    """ Tests that generated bounds are correct """
    points = [Marker(0, 0), Marker(-89.9, 180), Marker(90, -180)]
    bounds = Map(points)._autobounds()
    assert bounds['max_lat'] == 90
    assert bounds['min_lat'] == -89.9
    assert bounds['max_lon'] == 180
    assert bounds['min_lon'] == -180


def test_bounds_limits():
    """ Tests that too-large lats and lons are truncated to real bounds. """
    points = [Marker(0, 0), Marker(-190, 280), Marker(190, -280)]
    bounds = Map(points)._autobounds()
    assert bounds['max_lat'] == 90
    assert bounds['min_lat'] == -90
    assert bounds['max_lon'] == 180
    assert bounds['min_lon'] == -180


#########
# Color #
#########


def test_color_table(states):
    """ Tests that color can take a Table. """
    data = Table.read_table('tests/us-unemployment.csv')
    states.color(data).show()


def test_color_dict(states):
    """ Tests that color can take a dict. """
    data = Table.read_table('tests/us-unemployment.csv')
    states.color(dict(zip(*data.columns))).show()


def test_color_values_and_ids(states):
    """ Tests that color can take values and ids. """
    data = Table.read_table('tests/us-unemployment.csv')
    states.color(data['Unemployment'], data['State']).show()
