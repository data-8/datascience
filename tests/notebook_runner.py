import nbformat
import sys
from nbconvert.preprocessors import ExecutePreprocessor

with open(sys.argv[1]) as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600)

ep.preprocess(nb, {'metadata': {'path': 'tests/'}})