.PHONY: help docs serve_docs install test

DOCS_DIR = docs

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  install    to install the datascience package locally"
	@echo "  test       to run the tests"
	@echo "  docs       to build the docs"
	@echo "  clean_docs to remove the doc files"
	@echo "  serve_docs to serve the docs from a local Python server"

install:
	python setup.py develop

test:
	python setup.py test

docs:
	cd $(DOCS_DIR) ; make html

clean_docs:
	cd $(DOCS_DIR) ; make clean

serve_docs:
	cd $(DOCS_DIR)/_build/html ; python -m http.server
