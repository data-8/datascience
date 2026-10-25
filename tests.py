"""Convenience entry point for running the test suite.

Prefer invoking ``pytest`` directly (or ``make test``). The pytest options for
this project live in the ``[pytest]`` section of ``tox.ini``, so no arguments
need to be passed here.
"""

import sys

import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv[1:]))
