#!/bin/bash
set -e
source .venv/bin/activate

MYPYPATH=./typings mypy --ignore-missing-imports -p forager

export PYTHONPATH=forager
python3 -m pytest
