#!/bin/bash

# This script is used by Travis to automatically push the docs to the gh-pages
# branch of the datascience repo.

set -e # exit with nonzero exit code if anything fails

git config user.name "Travis CI"
git config user.email "travis@travis.com"

make docs

git add -A docs
git commit -m "[Travis] Build documentation"

make deploy_docs GH_REMOTE="https://${GH_TOKEN}@${GH_REPO}"
