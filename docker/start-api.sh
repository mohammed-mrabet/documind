#!/bin/sh
set -e
# Indexation des documents au démarrage (rapide : les morceaux existants sont remplacés, pas dupliqués).
python -m documind.ingest
exec uvicorn documind.api:app --host 0.0.0.0 --port 8000
