# Contributing

Thanks for your interest in contributing to this project! Below are quick guidelines to make it easy to get started.

1. Fork the repository and create a feature branch from `master`.
2. Write tests for any new functionality or bug fix.
3. Run the test suite locally before opening a PR:

   - Using `tox` (recommended):

     tox

   - Or with `pytest` directly after installing dev requirements:

     pip install -r requirements-tests.txt
     pip install -e .
     pytest

4. Keep changes focused and include a descriptive PR title and body.
5. Follow existing code style and add/update documentation as needed.

If you're unsure where to start, check the `issues` list for good first issues.

Thanks — maintainers
