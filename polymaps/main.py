# This code can be put in any Python module, it does not require IPython
# itself to be running already.  It only creates the magics subclass but
# doesn't instantiate it yet.
from __future__ import print_function
from IPython.core.magic import Magics, magics_class, line_cell_magic

# The class MUST call this class decorator at creation time
@magics_class
class Polymap(Magics):

	@line_cell_magic
	def polymap(self, line, cell=None):
		"""Polymap works both as %polymap and as %%polymap"""
		print("Full access to the main IPython object:", self.shell)
		print("Variables in the user namespace:", list(self.shell.user_ns.keys()))
		if cell is None:
			print("Called as line magic")
			return line
		else:
			print("Called as cell magic")
			return line, cell