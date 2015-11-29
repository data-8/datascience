import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand



with open('requirements.txt') as fid:
    install_requires = [l.strip() for l in fid.readlines() if l]

tests_requires = [
    'pytest',
    'coverage == 3.7.1',
    'coveralls == 0.5'
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
        self.pytest_args = []

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
    url = 'https://github.com/dsten/datascience',
    download_url = 'https://github.com/dsten/datascience/archive/%s.zip' % version,
    keywords = ['data', 'tools', 'berkeley'],
    classifiers = []
)
