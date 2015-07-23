from datascience import *
from unittest.mock import MagicMock, patch


def test_draw_map():
	""" Tests that draw_map returns HTML """
	region = MapRegion(geo_path='../data/us-states.json')
	map_html = draw_map((51.5135015, -0.1362392), regions=[region], zoom=3)
	assert isinstance(map_html, HTML)
	

def test_draw_marker():
	""" Tests that draw_map returns HTML """
	points = [MapPoint((51.5135015, -0.1358392))]
	map_html = draw_map((51.5135015, -0.1362392), points=points)
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