"""Dataset and request validation: schema, ranges, missing values, duplicates and target leakage."""
import math

import pandas as pd

from src.config import (
    FEATURES,
    FORBIDDEN_FEATURES,
    N_DUPLICATES,
    NUMERIC_FEATURES,
    RANGES,
    REQUIRED_COLUMNS,
    SEASONS,
    TARGET_CLF,
    TARGET_REG,
)


class DataValidationError(ValueError):
    """Raised when the dataset breaks a schema, range or leakage rule."""


class RecordValidationError(ValueError):
    """Raised when a single prediction record is invalid; .errors lists every problem."""

    def __init__(self, errors):
        self.errors = errors
        super().__init__("; ".join(f"{e['field']}: {e['problem']}" for e in errors))


def check_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    for col in NUMERIC_FEATURES + [TARGET_REG]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"Column '{col}' must be numeric.")


def check_ranges(df: pd.DataFrame) -> None:
    problems = []
    for col, (lo, hi) in RANGES.items():
        values = df[col].dropna()
        bad = int(((values < lo) | (values > hi)).sum())
        if bad:
            problems.append(f"{col}: {bad} value(s) outside [{lo}, {hi}]")
    if not df["season"].isin(SEASONS).all():
        problems.append("season: values other than A or B")
    if not df["cooperative_support"].isin([0, 1]).all():
        problems.append("cooperative_support: values other than 0 or 1")
    if not df[TARGET_CLF].isin([0, 1]).all():
        problems.append("high_risk: values other than 0 or 1")
    if (df[TARGET_REG] < 0.5 - 1e-9).any():
        problems.append("yield_tons: values below the 0.50 minimum")
    if df[TARGET_REG].isna().any() or df[TARGET_CLF].isna().any():
        problems.append("targets contain missing values")
    if problems:
        raise DataValidationError("Range check failed: " + "; ".join(problems))


def quality_report(df: pd.DataFrame) -> dict:
    """Count rows, missing values per column and exact duplicate rows."""
    missing = {c: int(df[c].isna().sum()) for c in df.columns}
    return {
        "rows": int(len(df)),
        "missing_by_column": missing,
        "missing_total": int(sum(missing.values())),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def expected_problems(base_rows: int) -> dict:
    """Problems the generator must have injected for a given base row count."""
    m = int(round(0.05 * base_rows))
    return {
        "rows": base_rows + N_DUPLICATES,
        "soil_moisture_pct_missing": m,
        "fertilizer_kg_missing": m,
        "duplicate_rows": N_DUPLICATES,
    }


def check_expected_problems(report: dict, base_rows: int) -> None:
    """Confirm the CSV carries exactly the deliberate problems (detects manual edits)."""
    exp = expected_problems(base_rows)
    got = {
        "rows": report["rows"],
        "soil_moisture_pct_missing": report["missing_by_column"]["soil_moisture_pct"],
        "fertilizer_kg_missing": report["missing_by_column"]["fertilizer_kg"],
        "duplicate_rows": report["duplicate_rows"],
    }
    if got != exp:
        raise DataValidationError(f"Dataset does not match the generator rules: expected {exp}, found {got}")


def assert_no_leakage(feature_columns) -> None:
    leaked = sorted(set(feature_columns) & FORBIDDEN_FEATURES)
    if leaked:
        raise DataValidationError(f"Target leakage: {leaked} cannot be used as input features.")


def validate_dataset(df: pd.DataFrame) -> dict:
    """Run schema, range and leakage checks and return the quality report."""
    check_schema(df)
    check_ranges(df)
    assert_no_leakage(FEATURES)
    return quality_report(df)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_record(record) -> dict:
    """Validate one prediction record; return a clean dict or raise RecordValidationError."""
    if not isinstance(record, dict):
        raise RecordValidationError([{"field": "record", "problem": "must be a JSON object"}])
    errors = []
    for name in FEATURES:
        if name not in record:
            errors.append({"field": name, "problem": "missing required field"})
    for name in record:
        if name not in FEATURES:
            errors.append({"field": name, "problem": "unknown field"})
    clean = {}
    for name in NUMERIC_FEATURES:
        if name not in record:
            continue
        value = record[name]
        lo, hi = RANGES[name]
        if not _is_number(value):
            errors.append({"field": name, "problem": "must be a number"})
        elif not lo <= value <= hi:
            errors.append({"field": name, "problem": f"must be between {lo} and {hi}"})
        else:
            clean[name] = float(value)
    if "season" in record:
        if record["season"] not in SEASONS or not isinstance(record["season"], str):
            errors.append({"field": "season", "problem": "must be 'A' or 'B'"})
        else:
            clean["season"] = record["season"]
    if "cooperative_support" in record:
        value = record["cooperative_support"]
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append({"field": "cooperative_support", "problem": "must be the integer 0 or 1"})
        elif value not in (0, 1):
            errors.append({"field": "cooperative_support", "problem": "must be 0 or 1"})
        else:
            clean["cooperative_support"] = value
    if errors:
        raise RecordValidationError(errors)
    return clean
