"""Load saved artifacts and score one validated record (shared by predict.py and app.py)."""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.config import DISCLAIMER, FEATURES, MODELS_DIR
from src.validation import validate_record


class ModelsNotFoundError(FileNotFoundError):
    """Raised when run_all.py has not produced the saved models yet."""


@lru_cache(maxsize=1)
def load_bundle(models_dir: str = str(MODELS_DIR)) -> dict:
    folder = Path(models_dir)
    needed = ["regression_model.joblib", "classification_model.joblib", "cluster_model.joblib", "metadata.json"]
    missing = [n for n in needed if not (folder / n).exists()]
    if missing:
        raise ModelsNotFoundError(f"Missing {missing} in {folder}. Run: python run_all.py --student-id <id> --seed <LLLL>")
    return {
        "regression": joblib.load(folder / "regression_model.joblib"),
        "classification": joblib.load(folder / "classification_model.joblib"),
        "cluster": joblib.load(folder / "cluster_model.joblib"),
        "metadata": json.loads((folder / "metadata.json").read_text(encoding="utf-8")),
    }


def predict_record(record: dict, bundle: dict | None = None) -> dict:
    """Validate then score one record. Raises RecordValidationError for bad input."""
    clean = validate_record(record)
    bundle = bundle or load_bundle()
    frame = pd.DataFrame([clean], columns=FEATURES)
    meta = bundle["metadata"]
    yield_estimate = float(bundle["regression"].predict(frame)[0])
    risk_class = int(bundle["classification"].predict(frame)[0])
    probability = float(bundle["classification"].predict_proba(frame)[0, 1])
    cluster = int(bundle["cluster"].predict(frame)[0])
    return {
        "verification_code": meta["verification_code"],
        "yield_estimate_tons": round(yield_estimate, 4),
        "risk_prediction": "high_risk" if risk_class == 1 else "not_high_risk",
        "risk_class": risk_class,
        "risk_probability": round(probability, 4),
        "cluster": cluster,
        "cluster_profile": meta["cluster_profiles"][str(cluster)]["profile"],
        "disclaimer": DISCLAIMER,
    }
