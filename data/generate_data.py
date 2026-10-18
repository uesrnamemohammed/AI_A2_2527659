"""Generate the individual Musanze potato dataset from the registration seed LLLL."""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import N_DUPLICATES, dataset_path, parse_identity  # noqa: E402


def base_row_count(seed: int) -> int:
    """Return 240 + (LLLL mod 41)."""
    return 240 + (int(seed) % 41)


def generate(seed: int) -> pd.DataFrame:
    """Create the dataset including deliberate missing values and exact duplicates."""
    rng = np.random.default_rng(int(seed))
    n = base_row_count(seed)

    rainfall = rng.uniform(600, 1400, n)
    temp = rng.uniform(12, 26, n)
    soil = rng.uniform(20, 80, n)
    fertilizer = rng.uniform(0, 250, n)
    field = rng.uniform(0.2, 4.0, n)
    season = rng.choice(["A", "B"], n)
    support = rng.integers(0, 2, n)
    noise = rng.normal(0.0, 0.80, n)

    season_effect = np.where(season == "A", 0.50, -0.20)
    yield_tons = (
        0.004 * rainfall
        + 0.06 * soil
        + 0.02 * fertilizer
        + 1.30 * field
        - 0.25 * np.abs(temp - 18)
        + 0.70 * support
        + season_effect
        + noise
    )
    yield_tons = np.clip(yield_tons, 0.50, None)

    yield_per_ha = yield_tons / field
    high_risk = (yield_per_ha < np.median(yield_per_ha)).astype(int)

    df = pd.DataFrame(
        {
            "rainfall_mm": rainfall,
            "avg_temp_c": temp,
            "soil_moisture_pct": soil,
            "fertilizer_kg": fertilizer,
            "field_size_ha": field,
            "season": season,
            "cooperative_support": support,
            "yield_tons": yield_tons,
            "high_risk": high_risk,
        }
    )

    m = int(round(0.05 * n))
    soil_positions = rng.choice(n, size=m, replace=False)
    fertilizer_positions = rng.choice(n, size=m, replace=False)
    df.iloc[soil_positions, df.columns.get_loc("soil_moisture_pct")] = np.nan
    df.iloc[fertilizer_positions, df.columns.get_loc("fertilizer_kg")] = np.nan

    duplicate_positions = rng.choice(n, size=N_DUPLICATES, replace=False)
    df = pd.concat([df, df.iloc[duplicate_positions]], ignore_index=True)
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write the CSV with four decimals and Unix newlines so the hash is platform independent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = df.to_csv(index=False, float_format="%.4f", lineterminator="\n")
    path.write_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    """SHA-256 of the exact CSV bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    identity = parse_identity(args.student_id, args.seed)
    path = dataset_path(identity["reg_digits"])
    df = generate(identity["seed"])
    write_csv(df, path)
    print(f"Verification code: {identity['verification_code']}")
    print(f"Wrote {path} with {len(df)} rows")
    print(f"SHA-256: {sha256_file(path)}")


if __name__ == "__main__":
    main()
