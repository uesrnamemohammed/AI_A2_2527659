"""Shared configuration, constants and identity rules for the Musanze Harvest Intelligence Service."""
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
DOCS_DIR = PROJECT_ROOT / "docs"

NUMERIC_FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
]
CATEGORICAL_FEATURES = ["season"]
BINARY_FEATURES = ["cooperative_support"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

TARGET_REG = "yield_tons"
TARGET_CLF = "high_risk"
REQUIRED_COLUMNS = FEATURES + [TARGET_REG, TARGET_CLF]
FORBIDDEN_FEATURES = {"yield_tons", "yield_per_ha", "high_risk"}

RANGES = {
    "rainfall_mm": (600.0, 1400.0),
    "avg_temp_c": (12.0, 26.0),
    "soil_moisture_pct": (20.0, 80.0),
    "fertilizer_kg": (0.0, 250.0),
    "field_size_ha": (0.2, 4.0),
}
SEASONS = ("A", "B")

TEST_SIZE = 0.25
CV_FOLDS = 5
CLUSTER_K_RANGE = range(2, 7)
N_DUPLICATES = 3

DISCLAIMER = (
    "Classroom decision-support prototype trained on synthetic data. "
    "It does not replace an agronomist and must not be used to decide access to "
    "finance, insurance or public services. Human review is required."
)


class IdentityError(ValueError):
    """Raised when the student id or seed breaks the individual-seed rule."""


def parse_identity(student_id: str, seed: int) -> dict:
    """Derive registration digits and the verification code, rejecting a seed that is not LLLL."""
    local_part = str(student_id).split("@")[0]
    digits = re.sub(r"\D", "", local_part)
    if len(digits) < 4:
        raise IdentityError(
            f"Student id '{student_id}' must contain the registration digits (at least 4 digits)."
        )
    last4 = digits[-4:]
    expected_seed = int(last4)
    if int(seed) != expected_seed:
        raise IdentityError(
            f"Seed {seed} rejected: the seed must equal LLLL = {expected_seed} "
            f"(last four digits of the registration number in '{student_id}')."
        )
    return {
        "student_id": student_id,
        "reg_digits": digits,
        "seed": expected_seed,
        "verification_code": f"AI-A2-{last4}",
    }


def dataset_path(reg_digits: str) -> Path:
    return DATA_DIR / f"AI_A2_{reg_digits}.csv"
