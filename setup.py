import sys
import os
import IPython
from setuptools import setup
from setuptools.command.test import test as TestCommand

install_requires = [
  'numpy',
  'matplotlib',
  'pandas',
  'folium'
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


setup(
  name = 'datascience',
  packages = ['datascience'],
  version = '0.2.4',
  install_requires = install_requires,
  tests_require = test_requires,
  cmdclass = {'test': PyTest},
  description = 'A Jupyter notebook Python library for introductory data science',
  author = 'John DeNero, David Culler, Alvin Wan, Sam Lau',
  author_email = 'ds8-instructors@berkeley.edu',
  url = 'https://github.com/dsten/datascience',
  download_url = 'https://github.com/dsten/datascience/archive/0.2.3.zip',
  keywords = ['data', 'tools', 'berkeley'],
  classifiers = [],
)
