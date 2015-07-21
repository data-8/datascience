from IPython.display import display_javascript

__maps_js_has_initialized__ = False

# Uses a hack to load the Javascript needed to render the map. Becaose of this,
# calls to draw_map cannot be in the same cell as the import statement.
#
# Maybe it's a good idea to put this login in a line magic?
def init_js():
    global __maps_js_has_initialized__
    if not __maps_js_has_initialized__:
        # Keep in sync with the the data_files variable in setup.py
        display_javascript("IPython.load_extensions('datascience_js/maps')")
