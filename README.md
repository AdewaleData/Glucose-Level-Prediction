# Glucose-Level-Prediction

Non-invasive glucose regression from PPG/vitals: notebook and trained models in `model/`, data in `dataset/`, API in `api/`.

Run API locally (from repo root): `pip install -r api/requirements.txt` then `python -m uvicorn api.main:app --host 127.0.0.1 --port 8000` or `.\run_api.ps1`.
