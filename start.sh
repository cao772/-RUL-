#!/usr/bin/env bash
set -e
PYTHON=${PYTHON:-python3}
if [ ! -d .venv ]; then "$PYTHON" -m venv .venv; fi
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py
