import sys
from distutils.core import setup
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


setup(
  name = 'datatools',
  packages = ['datatools'],
  version = '0.1',
  install_requires = install_requires,
  tests_require = test_requires,
  cmdclass = {'test': PyTest},
  description = 'datatools for UC Berkeley\'s DS10 course',
  author = 'John DeNero, David Culler, Alvin Wan',
  author_email = '',
  url = 'https://github.com/alvinwan/datatools',
  download_url = 'https://github.com/github/alvinwan/tarball/0.1',
  keywords = ['data', 'tools', 'berkeley'],
  classifiers = [],
)
