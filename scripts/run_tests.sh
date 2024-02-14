#!/bin/bash
set -e

flake8 forager/ tests/
MYPYPATH=./typings mypy --ignore-missing-imports -p forager

export PYTHONPATH=forager
python3 -m pytest
