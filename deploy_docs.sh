#!/bin/bash

# This script is used by Travis to automatically build and push the docs to the
# gh-pages branch of the datascience repo.

set -e # exit with nonzero exit code if anything fails

# Only build docs on master branch
if [[ $TRAVIS_BRANCH != 'master' ]]; then
  echo "Not building docs since we're not on the master branch."
  exit 0
fi

git config user.name "Travis CI"
git config user.email "travis@travis.com"

make docs

git add -A docs
git commit -m "[Travis] Build documentation"

make deploy_docs GH_REMOTE="https://${GH_TOKEN}@${GH_REPO}"
