# Francis Agent

Local autonomy loop with tools (files, gitops, httpjson, kvstore, search, shell, web).

## Quickstart
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\francis.py

## Dev
pre-commit install
pre-commit run --all-files
pytest -q
