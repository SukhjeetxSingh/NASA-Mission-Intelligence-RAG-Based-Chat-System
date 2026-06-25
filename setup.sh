#!/usr/bin/env bash
set -e

python3.12 -m venv ragas_env
source ragas_env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import ragas; print('ragas ok')"
