"""FastAPI service: uvicorn app:app --host 127.0.0.1 --port 8000"""
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.config import DISCLAIMER
from src.inference import ModelsNotFoundError, load_bundle, predict_record
from src.validation import RecordValidationError

app = FastAPI(
    title="Musanze Harvest Intelligence Service",
    description=DISCLAIMER,
    version="1.0.0",
)


class FarmRecord(BaseModel):
    """One farm record. Unknown fields, wrong types and out-of-range values are rejected with HTTP 422."""

    model_config = ConfigDict(extra="forbid")

    rainfall_mm: float = Field(..., strict=True, ge=600, le=1400)
    avg_temp_c: float = Field(..., strict=True, ge=12, le=26)
    soil_moisture_pct: float = Field(..., strict=True, ge=20, le=80)
    fertilizer_kg: float = Field(..., strict=True, ge=0, le=250)
    field_size_ha: float = Field(..., strict=True, ge=0.2, le=4.0)
    season: Literal["A", "B"]
    cooperative_support: Literal[0, 1]


def _bundle():
    try:
        return load_bundle()
    except ModelsNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/health")
def health() -> dict:
    try:
        code = load_bundle()["metadata"]["verification_code"]
        return {"status": "ok", "models_loaded": True, "verification_code": code}
    except ModelsNotFoundError:
        return {"status": "degraded", "models_loaded": False, "verification_code": None}


@app.get("/model-info")
def model_info() -> dict:
    meta = _bundle()["metadata"]
    keys = [
        "verification_code", "student_id", "seed", "dataset_sha256", "features", "ranges",
        "regression_model", "classification_model", "selected_k",
        "regression_test", "classification_test", "synthetic_data", "disclaimer",
    ]
    return {k: meta[k] for k in keys}


@app.post("/predict")
def predict(record: FarmRecord) -> dict:
    bundle = _bundle()
    try:
        return predict_record(record.model_dump(), bundle)
    except RecordValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors)
