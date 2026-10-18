"""Automated tests: valid inference, failure paths, reproducibility, leakage and the API."""
import copy
import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.generate_data import base_row_count, generate, sha256_file, write_csv  # noqa: E402
from src import config  # noqa: E402
from src.inference import load_bundle, predict_record  # noqa: E402
from src.validation import (  # noqa: E402
    DataValidationError,
    RecordValidationError,
    assert_no_leakage,
    quality_report,
    validate_record,
)

STUDENT_ID = "ug2527659@ines.ac.rw"
SEED = 7659
VALID = {
    "rainfall_mm": 1000.0,
    "avg_temp_c": 18.5,
    "soil_moisture_pct": 55.0,
    "fertilizer_kg": 120.0,
    "field_size_ha": 1.5,
    "season": "A",
    "cooperative_support": 1,
}


def _ensure_models():
    if not (config.MODELS_DIR / "metadata.json").exists():
        import run_all

        run_all.run(STUDENT_ID, SEED, make_report=False, verbose=False)
    return load_bundle()


def test_valid_inference_returns_all_fields():
    result = predict_record(VALID, _ensure_models())
    assert result["verification_code"] == "AI-A2-7659"
    assert result["yield_estimate_tons"] > 0
    assert result["risk_prediction"] in ("high_risk", "not_high_risk")
    assert 0.0 <= result["risk_probability"] <= 1.0
    assert isinstance(result["cluster"], int)


def test_missing_field_is_rejected():
    record = copy.deepcopy(VALID)
    del record["rainfall_mm"]
    with pytest.raises(RecordValidationError) as exc:
        validate_record(record)
    assert any(e["field"] == "rainfall_mm" for e in exc.value.errors)


def test_wrong_type_is_rejected():
    record = copy.deepcopy(VALID)
    record["avg_temp_c"] = "warm"
    with pytest.raises(RecordValidationError) as exc:
        validate_record(record)
    assert any(e["field"] == "avg_temp_c" and "number" in e["problem"] for e in exc.value.errors)


def test_range_error_is_rejected():
    record = copy.deepcopy(VALID)
    record["rainfall_mm"] = 2500.0
    record["season"] = "C"
    with pytest.raises(RecordValidationError) as exc:
        validate_record(record)
    fields = {e["field"] for e in exc.value.errors}
    assert {"rainfall_mm", "season"} <= fields


def test_dataset_is_reproducible_and_matches_fingerprint():
    first, second = generate(SEED), generate(SEED)
    pd.testing.assert_frame_equal(first, second)
    target = ROOT / "artifacts" / "_repro_check.csv"
    write_csv(first, target)
    digest = sha256_file(target)
    target.unlink()
    assert digest == hashlib.sha256(
        first.to_csv(index=False, float_format="%.4f", lineterminator="\n").encode("utf-8")
    ).hexdigest()
    fingerprint_file = config.ARTIFACTS_DIR / "data_fingerprint.txt"
    if fingerprint_file.exists():
        assert f"sha256: {digest}" in fingerprint_file.read_text(encoding="utf-8")


def test_dataset_signature_and_deliberate_problems():
    df = generate(SEED)
    base = base_row_count(SEED)
    assert base == 273 and len(df) == base + 3
    report = quality_report(df)
    assert report["missing_by_column"]["soil_moisture_pct"] == 14
    assert report["missing_by_column"]["fertilizer_kg"] == 14
    assert report["duplicate_rows"] == 3


def test_wrong_seed_is_rejected():
    with pytest.raises(config.IdentityError):
        config.parse_identity(STUDENT_ID, 1234)
    assert config.parse_identity(STUDENT_ID, SEED)["verification_code"] == "AI-A2-7659"


def test_target_leakage_is_blocked():
    with pytest.raises(DataValidationError):
        assert_no_leakage(config.FEATURES + ["yield_tons"])
    assert_no_leakage(config.FEATURES)


def test_preprocessing_is_fitted_on_training_data_only():
    from src.preprocessing import build_preprocessor, clean_data, split_data

    cleaned = clean_data(generate(SEED))
    X_train, X_test, *_ = split_data(cleaned, SEED)
    pre = build_preprocessor().fit(X_train)
    imputer = pre.named_transformers_["num"].named_steps["imputer"]
    train_medians = X_train[config.NUMERIC_FEATURES].median().to_numpy()
    assert (abs(imputer.statistics_ - train_medians) < 1e-9).all()
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_api_endpoints_and_validation_errors():
    from fastapi.testclient import TestClient

    _ensure_models()
    from app import app

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200 and health.json()["verification_code"] == "AI-A2-7659"
    info = client.get("/model-info")
    assert info.status_code == 200 and info.json()["synthetic_data"] is True
    ok = client.post("/predict", json=VALID)
    assert ok.status_code == 200 and ok.json()["verification_code"] == "AI-A2-7659"
    bad = client.post("/predict", json={**VALID, "rainfall_mm": 5000})
    assert bad.status_code == 422
    missing = {k: v for k, v in VALID.items() if k != "season"}
    assert client.post("/predict", json=missing).status_code == 422
    wrong_type = client.post("/predict", json={**VALID, "fertilizer_kg": "lots"})
    assert wrong_type.status_code == 422
