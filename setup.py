import sys
import os
import IPython
from setuptools import setup
from setuptools.command.test import test as TestCommand

install_requires = [
  'numpy',
  'matplotlib',
  'pandas'
]

test_requires = [
  'pytest'
]

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


# Installs GMaps Javascript in the nbextensions folder for loading.
# We load this file using IPython.load_extensions('datascience_js/maps') in
# Javascript. Keep in sync with the path in maps/leader.py
ipython_dir = IPython.utils.path.get_ipython_dir()
data_files = [(os.path.join(ipython_dir, "nbextensions/datascience_js"),
               ["datascience/maps/js/maps.js"] )]

setup(
  name = 'datascience',
  py_modules = ['datascience'],
  version = '0.2.1',
  install_requires = install_requires,
  tests_require = test_requires,
  data_files = data_files,
  cmdclass = {'test': PyTest},
  description = 'A Python library for introductory data science',
  author = 'John DeNero, David Culler, Alvin Wan, Sam Lau',
  author_email = 'ds-instr@berkeley.edu',
  url = 'https://github.com/dsten/datascience',
  download_url = 'https://github.com/dsten/datascience/archive/0.2.0.zip',
  keywords = ['data', 'tools', 'berkeley'],
  classifiers = [],
)
