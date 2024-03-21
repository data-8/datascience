Maps (``datascience.maps``)
===========================

.. automodule:: datascience.maps

**Summary of Methods for FoliumWrapper**

The folium wrapper class creates a map element that can be drawn. It takes an abstract base class as an argument.
    
Display

.. autosummary::
    :toctree: _autosummary
    
    _FoliumWrapper.draw
    _FoliumWrapper.as_html
    _FoliumWrapper.show
    
    
Edit
   
.. autosummary::
    :toctree: _autosummary
    
    _FoliumWrapper._inline_map
    _FoliumWrapper._set_folium_map
    

**Summary of Methods for Map**

The Map class takes in list of features and ids along with width and height as arguments.

Creation

.. autosummary::
    :toctree: _autosummary
    
    Map.__init__
    Map.copy
    Map._create_map
    
    
Accessing Values

.. autosummary::
    :toctree: _autosummary
    
    Map.__getItem__
    Map.__len__
    Map.__iter__
    Map._autozoom
    Map._autobounds
    Map.features
    Map.read_geojson
    Map._read_geojson_features
    
    
Mutation

.. autosummary::
    :toctree: _autosummary
    
    Map._set_folium_wrap
    Map.format
    Map.geojson
    Map.color
    
Transformation

.. autosummary::
    :toctree: _autosummary
    
    Map.overlay
    
**Summary of Marker Class**

The Marker class instantiates a marker that is displayed with _FoliumWrapper's simple_marker method. The color of the marker can either be chosen from [‘red’, ‘blue’, ‘green’, ‘purple’, ‘orange’, ‘darkred’,'lightred’, ‘beige’, ‘darkblue’, ‘darkgreen’, ‘cadetblue’, ‘darkpurple’, 
‘white’, ‘pink’, ‘lightblue’, ‘lightgreen’, ‘gray’, ‘black’, ‘lightgray’]  or using a hexcode. 


Creation

.. autosummary::
    :toctree: _autosummary
    
    Marker.__init__
    Marker.copy
    
Mutation

.. autosummary::
    :toctree: _autosummary
    
    Marker.geojson
    Marker.format
    Marker.draw_on
    Marker.map
    Marker.map_table
    
Accessing Values

.. autosummary::
    :toctree: _autosummary
    
    Marker.lat_lons
    
**Summary of Region Class**

Used to display a GeoJSON feature when using Folium's geo_json method. 

Creation 

.. autosummary::
    :toctree: _autosummary
    
    Region.__init__
    Region.copy
    
Mutation

.. autosummary::
    :toctree: _autosummary
    
    Region.geojson
    Region.format
    Region.draw_on
    
    
Accessing values

.. autosummary::
    :toctree: _autosummary
    
    Region.lat_lons
    Region.type
    Region.polygons
    Region.properties
    Region._lat_lons_from_geojson
    Region.get_coordinates

