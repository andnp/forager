#!/bin/bash
set -e

MYPYPATH=./typings mypy --ignore-missing-imports -p forager

export PYTHONPATH=forager
python3 -m pytest
