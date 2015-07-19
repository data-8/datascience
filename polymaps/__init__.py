from .main import Polymap
from IPython.display import Javascript, display


def load_ipython_extension(ipython):
	# The `ipython` argument is the currently active `InteractiveShell`
	# instance, which can be used in any way. This allows you to register
	# new magics or aliases, for example.
	ipython.register_magics(Polymap)
	
	# Loads the JS extension.
	display(Javascript("""
		IPython.load_extensions('datascience_polymaps_js/polymaps_views')
		"""))