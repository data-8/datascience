import sys
from setuptools import setup


if sys.version_info < (3, 6):
    raise ValueError('This package requires python >= 3.6')

with open('requirements.txt') as fid:
    install_requires = [l.strip() for l in fid.readlines() if l]

with open('datascience/version.py') as fid:
     for line in fid:
         if line.startswith('__version__'):
             version = line.strip().split()[-1][1:-1]
             break

# Test dependencies are listed in requirements-tests.txt and the test suite is
# run with pytest directly (`make test`). The `setup.py test` command was
# removed in Setuptools 72, so it is no longer wired up here.

setup(
    name = 'datascience',
    packages = ['datascience'],
    python_requires='>=3.6',
    version = version,
    install_requires = install_requires,
    description = 'A Jupyter notebook Python library for introductory data science',
    long_description = 'A Jupyter notebook Python library for introductory data science',
    author = 'John DeNero, David Culler, Alvin Wan, Sam Lau',
    author_email = 'ds8-instructors@berkeley.edu',
    url = 'https://github.com/data-8/datascience',
    download_url = 'https://github.com/data-8/datascience/archive/%s.zip' % version,
    keywords = ['data', 'tools', 'berkeley'],
    classifiers = [],
    package_data={"datascience": ["geodata/*.csv"]}
)
