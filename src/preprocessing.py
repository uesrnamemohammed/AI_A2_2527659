"""Preprocessing: one ColumnTransformer shared by training and inference, plus the fixed split."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET_CLF,
    TARGET_REG,
    TEST_SIZE,
)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows (keeping the first). Imputation happens later, inside the pipeline."""
    return df.drop_duplicates(keep="first")


def build_preprocessor() -> ColumnTransformer:
    """Median imputation and scaling for numbers, one-hot for season, passthrough for the binary flag."""
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        sparse_threshold=0.0,
    )


def split_data(df: pd.DataFrame, seed: int):
    """Fixed stratified train/test split with random_state=LLLL; the index keeps the original row id."""
    X = df[FEATURES]
    return train_test_split(
        X,
        df[TARGET_REG],
        df[TARGET_CLF],
        test_size=TEST_SIZE,
        random_state=int(seed),
        stratify=df[TARGET_CLF],
    )
