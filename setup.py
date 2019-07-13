import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand


if sys.version_info < (3, 0):
    raise ValueError('This package requires python >= 3.0')

with open('requirements.txt') as fid:
    install_requires = [l.strip() for l in fid.readlines() if l]

tests_requires = [
    'pytest',
    'coverage==4.5.3',
    'coveralls'
]


with open('datascience/version.py') as fid:
    for line in fid:
        if line.startswith('__version__'):
            version = line.strip().split()[-1][1:-1]
            break

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ['tests']

    def finalize_options(self):
        TestCommand.finalize_options(self)

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name = 'datascience',
    packages = ['datascience'],
    version = version,
    install_requires = install_requires + tests_requires,
    tests_require = tests_requires,
    cmdclass = {'test': PyTest},
    description = 'A Jupyter notebook Python library for introductory data science',
    author = 'John DeNero, David Culler, Alvin Wan, Sam Lau',
    author_email = 'ds8-instructors@berkeley.edu',
    url = 'https://github.com/data-8/datascience',
    download_url = 'https://github.com/data-8/datascience/archive/%s.zip' % version,
    keywords = ['data', 'tools', 'berkeley'],
    classifiers = []
)
