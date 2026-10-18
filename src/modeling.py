"""Candidate models, cross-validated selection, held-out evaluation and clustering."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import CLUSTER_K_RANGE, CV_FOLDS, NUMERIC_FEATURES
from src.preprocessing import build_preprocessor

BASELINE_REG = "mean_baseline"
BASELINE_CLF = "majority_baseline"


def _pipe(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def regression_candidates(seed: int) -> dict:
    return {
        BASELINE_REG: _pipe(DummyRegressor(strategy="mean")),
        "linear_regression": _pipe(LinearRegression()),
        "random_forest": _pipe(RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1)),
        "gradient_boosting": _pipe(GradientBoostingRegressor(random_state=seed)),
    }


def classification_candidates(seed: int) -> dict:
    return {
        BASELINE_CLF: _pipe(DummyClassifier(strategy="most_frequent")),
        "logistic_regression": _pipe(LogisticRegression(max_iter=1000, random_state=seed)),
        "random_forest": _pipe(RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)),
        "gradient_boosting": _pipe(GradientBoostingClassifier(random_state=seed)),
    }


def regression_scores(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def run_regression(X_train, y_train, X_test, y_test, seed: int):
    """Compare the baseline and candidates; select by training cross-validation RMSE, report test scores."""
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    results, fitted = {}, {}
    for name, pipe in regression_candidates(seed).items():
        cv_rmse = float(
            -cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error").mean()
        )
        pipe.fit(X_train, y_train)
        results[name] = {"cv_rmse_train": cv_rmse, "test": regression_scores(y_test, pipe.predict(X_test))}
        fitted[name] = pipe
    candidates = [n for n in results if n != BASELINE_REG]
    selected = min(candidates, key=lambda n: results[n]["cv_rmse_train"])
    return results, fitted, selected


def classification_scores(y_true, y_pred, proba) -> dict:
    labels = [0, 1]
    valid_auc = len(set(np.asarray(y_true).tolist())) == 2
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "roc_auc": float(roc_auc_score(y_true, proba)) if valid_auc else None,
    }


def run_classification(X_train, y_train, X_test, y_test, seed: int):
    """Compare the baseline and candidates; select by training cross-validation F1, report test scores."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    results, fitted = {}, {}
    for name, pipe in classification_candidates(seed).items():
        cv_f1 = float(cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1").mean())
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = {"cv_f1_train": cv_f1, "test": classification_scores(y_test, pred, proba)}
        fitted[name] = pipe
    candidates = [n for n in results if n != BASELINE_CLF]
    selected = max(candidates, key=lambda n: results[n]["cv_f1_train"])
    return results, fitted, selected


def _describe(feature: str, z: float) -> str:
    words = {
        "rainfall_mm": ("low rainfall", "high rainfall"),
        "avg_temp_c": ("cool temperature", "warm temperature"),
        "soil_moisture_pct": ("dry soil", "moist soil"),
        "fertilizer_kg": ("little fertilizer", "heavy fertilizer"),
        "field_size_ha": ("small fields", "large fields"),
    }
    low, high = words[feature]
    return high if z > 0 else low


def run_clustering(X_train, seed: int):
    """Scale features, test K=2..6 with silhouette, fit the selected K and describe each cluster."""
    pre = build_preprocessor()
    Z = pre.fit_transform(X_train)
    scores = {}
    for k in CLUSTER_K_RANGE:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
        scores[k] = float(silhouette_score(Z, labels))
    best_k = max(scores, key=scores.get)

    model = Pipeline(
        [("prep", build_preprocessor()), ("kmeans", KMeans(n_clusters=best_k, n_init=10, random_state=seed))]
    ).fit(X_train)
    labels = model.predict(X_train)

    overall_mean = X_train[NUMERIC_FEATURES].mean()
    overall_std = X_train[NUMERIC_FEATURES].std(ddof=0)
    profiles = {}
    for c in range(best_k):
        rows = X_train[labels == c]
        means = rows[NUMERIC_FEATURES].mean()
        z = ((means - overall_mean) / overall_std).to_dict()
        notable = sorted(
            [(f, v) for f, v in z.items() if abs(v) >= 0.4], key=lambda t: abs(t[1]), reverse=True
        )[:3]
        traits = [_describe(f, v) for f, v in notable]
        season_a = float((rows["season"] == "A").mean())
        support = float(rows["cooperative_support"].mean())
        if season_a >= 0.7:
            traits.append("mostly season A")
        elif season_a <= 0.3:
            traits.append("mostly season B")
        if support >= 0.7:
            traits.append("mostly supported by the cooperative")
        elif support <= 0.3:
            traits.append("mostly without cooperative support")
        summary = ", ".join(traits) if traits else "close to the average on every feature"
        profiles[str(c)] = {
            "size_train": int(len(rows)),
            "mean_features": {f: float(means[f]) for f in NUMERIC_FEATURES},
            "share_season_A": season_a,
            "share_cooperative_support": support,
            "profile": f"Farms with {summary}.",
        }
    return scores, best_k, model, profiles
