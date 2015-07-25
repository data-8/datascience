from datascience import *
from unittest.mock import MagicMock, patch
import pytest
import copy

#########
# Utils #
#########


@pytest.fixture(scope='function')
def locations():
	# locations are a list of list of lat-long pairs, for regions with multiple polygons (i.e., Alaska)
	locations1 = [[(51.5135015, -0.1358392), (51.5137, -0.1358392), (51.5132, -0.138)], 
	              [(51.514, -0.1361), (51.5143, -0.1361), (51.5145, -0.1383)]]
	return copy.deepcopy(locations1)


@pytest.fixture(scope='function')
def points():
	# points are just a list of lat-long pairs
	locations2 = [(51.514, -0.132), (51.5143, -0.132), (51.5145, -0.135)]
	return locations2[:]


@pytest.fixture(scope='function')
def region1(locations):
	region1 = MapRegion(locations=locations, geo_formatted=True)
	return region1


@pytest.fixture(scope='function')
def region2(points):
	region2 = MapRegion(points=points, geo_formatted=True)
	return region2


############
# Overview #
############


def test_draw_map():
	""" Tests that draw_map returns HTML """
	region = MapRegion(geo_path='../data/us-states.json')
	map_html = draw_map((40, -90), regions=[region], zoom_start=4)
	assert isinstance(map_html, HTML)


def test_setup_map():
	""" Tests that to_html() returns HTML """
	center = [45.5244, -122.6699]
	kwargs = {
		'tiles': 'Stamen Toner',            
		'zoom_start': 17,                   
		'width': 960,
		'height': 500,
		'points': [],                       
		'regions': []                       
	}
	map_html = Map(location=center, **kwargs).map().to_html()
	assert isinstance(map_html, HTML)
	
	
def test_draw_map_arg(region1):
	""" Tests that draw_map accepts a MapRegion or MapPoint as the first arg """
	map_html = draw_map(region1)
	assert isinstance(map_html, HTML)
	
	point = MapPoint((51.5135015, -0.1358392))
	map_html = draw_map(point)
	assert isinstance(map_html, HTML)


############
# MapPoint #
############


def test_draw_marker():
	""" Tests that draw_map returns HTML """
	points = [MapPoint((51.5135015, -0.1358392))]
	map_html = draw_map((51.5135015, -0.1362392), points=points)
	assert isinstance(map_html, HTML)


def test_point_init():
	""" Tests that you can pass in a collection of coordinates or two coordinates"""
	points = [MapPoint((51.5135015, -0.1358392)), MapPoint(51.5135015, -0.1358392)]
	map_html = draw_map((51.5135015, -0.1362392), points=points)
	assert isinstance(map_html, HTML)


#############
# MapRegion #
#############


def test_map_geo_str(region1, region2):
	map_html = draw_map((51.5135015, -0.1362392), regions=[region1, region2])
	assert isinstance(map_html, HTML)


def test_group_recursive_jsons(region1, region2):
	""" Tests that recurisve _to_json returns a flat layer of geoJSONs """
	papaRegion = MapRegion(regions=[region1, region2])
	grandpaRegion = MapRegion(regions=[papaRegion])

	grandpaRegion['composite'] = MapRegion.GROUP
	papaRegion['composite'] = MapRegion.GROUP
	json_data = grandpaRegion.to_feature()
	assert isinstance(json_data, list)
	assert len(json_data) == 2

	json_data = grandpaRegion.to_polygon()
	assert isinstance(json_data, list)
	assert len(json_data) == 3


def test_group_to_json(region1, region2, locations, points):
	""" Tests that _to_json returns the correct data """
	
	grab_locations = lambda json_data: json_data['features'][0]['geometry']['coordinates']

	json_data = region1._to_json()
	data = grab_locations(json_data)
	assert data == locations

	json_data = region2._to_json()
	data = grab_locations(json_data)
	assert data == [points]
	
	
def test_long_lat_flip(locations, points):
	""" Tests that MapRegion undoes geo_format to comply with EPSG standard, by default """
	region1 = MapRegion(locations=locations)
	region2 = MapRegion(points=points)

	grab_locations = lambda json_data: json_data['features'][0]['geometry']['coordinates']

	json_data = region1._to_json()
	data = grab_locations(json_data)
	assert data == [[(y, x) for x, y in location] for location in locations]

	json_data = region2._to_json()
	data = grab_locations(json_data)
	assert data == [[(y, x) for x, y in points]]


############
# Features #
############


def test_map_autofit_points():
	""" Test that autofit doesn't break with points """
	points = [MapPoint((51.5135015, -0.1358392))]
	map_html = draw_map(points=points)
	assert isinstance(map_html, HTML)


def test_map_autofit_regions(region1):
	""" Test that autofit doesn't break with regions """
	map_html = draw_map(regions=[region1])
	assert isinstance(map_html, HTML)
	
	
def test_map_autofit_mix(region1):
	""" Test that autofit doesn't break with a mix of both regions and points """
	points = [MapPoint((51.5135015, -0.1358392))]
	map_html = draw_map(points=points, regions=[region1])
	assert isinstance(map_html, HTML)
	

###############
# TEST BOUNDS #
###############


def test_bounds():
	""" Tests that generated bounds are correct """
	points = [MapPoint((0, 0)), MapPoint((-90, 180)), MapPoint((90, -180))]
	bounds = Map(points=points)._autobounds()
	assert bounds['max_lat'] == 90
	assert bounds['min_lat'] == -90
	assert bounds['max_long'] == 180
	assert bounds['min_long'] == -180
	
	
def test_bounds_limits():
	""" 
	Tests that randomly large lats and lots are truncated to 
	the correct max, min values """

	points = [MapPoint((0, 0)), MapPoint((-100, 200)), MapPoint((90, -180))]
	bounds = Map(points=points)._autobounds()
	assert bounds['max_lat'] == 90
	assert bounds['min_lat'] == -90
	assert bounds['max_long'] == 180
	assert bounds['min_long'] == -180
	
	
#######################
# Additional Features #
#######################

def test_data_abstraction():
	""" Tests that data behaves normally """
	data = Data('data/us-states.json')
	map_html = draw_map(data.CA)
	assert isinstance(map_html, HTML)