"""
Run from repo root:
  pip install -r api/requirements.txt
  uvicorn api.main:app --host 0.0.0.0 --port 8000

The .pkl was saved with scikit-learn 1.6.1; use that version on the server
or the pipeline may fail to load (see api/requirements.txt).
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.features import BASE_INPUT_COLUMNS, add_engineered_features

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "model" / "best_glucose_model_tuned.pkl"
MODEL_PATH = Path(os.environ.get("GLUCOSE_MODEL_PATH", str(DEFAULT_MODEL)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield


app = FastAPI(title="Glucose prediction API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.is_file():
            raise RuntimeError(f"Model file not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


class PredictRequest(BaseModel):
    Gender: float | int = Field(..., description="0/1 or encoded sex")
    Age: float
    Height: float
    Weight: float
    BMI: float
    Peak_Diff: float
    index: float
    PPG_Signal: float
    Heart_Rate: float
    Pulse_Area: float
    Systolic_Peak: float
    Diastolic_Peak: float


class PredictResponse(BaseModel):
    glucose_level: float


class BatchPredictRequest(BaseModel):
    rows: list[PredictRequest]


class BatchPredictResponse(BaseModel):
    glucose_level: list[float]


def _records_to_frame(rows: list[PredictRequest]) -> pd.DataFrame:
    data = [r.model_dump() for r in rows]
    df = pd.DataFrame(data)
    missing = set(BASE_INPUT_COLUMNS) - set(df.columns)
    if missing:
        raise HTTPException(400, f"missing columns: {sorted(missing)}")
    df = df[BASE_INPUT_COLUMNS]
    df = add_engineered_features(df)
    pipe = get_model()
    names = list(pipe.feature_names_in_)
    for n in names:
        if n not in df.columns:
            raise HTTPException(500, f"internal: expected column {n!r} after feature build")
    return df[names]


@app.get("/health")
def health():
    return {"ok": True, "model_path": str(MODEL_PATH.resolve())}


@app.post("/v1/predict", response_model=PredictResponse)
def predict_one(body: PredictRequest):
    X = _records_to_frame([body])
    y = get_model().predict(X)
    return PredictResponse(glucose_level=float(y[0]))


@app.post("/v1/predict_batch", response_model=BatchPredictResponse)
def predict_batch(body: BatchPredictRequest):
    if not body.rows:
        raise HTTPException(400, "rows must be non-empty")
    X = _records_to_frame(body.rows)
    y = get_model().predict(X)
    return BatchPredictResponse(glucose_level=[float(v) for v in y])
