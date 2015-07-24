from datascience import *
from unittest.mock import MagicMock, patch
import pytest

#########
# Utils #
#########


@pytest.fixture(scope='function')
def locations1():
	# locations are a list of list of lat-long pairs, for regions with multiple polygons (i.e., Alaska)
	locations1 = [[(51.5135015, -0.1358392), (51.5137, -0.1358392), (51.5132, -0.138)], 
				[(51.514, -0.1361), (51.5143, -0.1361), (51.5145, -0.1383)]]
	return locations1


@pytest.fixture(scope='function')
def locations2():
	# points are just a list of lat-long pairs
	locations2 = [(51.514, -0.132), (51.5143, -0.132), (51.5145, -0.135)]
	return locations2


@pytest.fixture(scope='function')
def region1(locations1):
	region1 = MapRegion(locations=locations1)
	return region1


@pytest.fixture(scope='function')
def region2(locations2):
	region2 = MapRegion(points=locations2)
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


############
# MapPoint #
############


def test_draw_marker():
	""" Tests that draw_map returns HTML """
	points = [MapPoint((51.5135015, -0.1358392))]
	map_html = draw_map((51.5135015, -0.1362392), points=points)
	assert isinstance(map_html, HTML)


#############
# MapRegion #
#############


def test_map_geo_str(region1, region2):
	map_html = draw_map((51.5135015, -0.1362392), regions=[region1, region2])
	assert isinstance(map_html, HTML)


def test_group_recursive_jsons(region1, region2, locations1, locations2):
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


def test_group_to_json(region1, region2, locations1, locations2):
	""" Tests that _to_json returns the correct data """
	
	grab_locations = lambda json_data: json_data['features'][0]['geometry']['coordinates']

	json_data = region1._to_json()
	locations = grab_locations(json_data)
	assert locations == locations1

	json_data = region2._to_json()
	locations = grab_locations(json_data)
	assert locations == [locations2]
