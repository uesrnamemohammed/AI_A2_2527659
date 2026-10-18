"""One-command pipeline: python run_all.py --student-id <id> --seed <LLLL>"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from data.generate_data import base_row_count, generate, sha256_file, write_csv  # noqa: E402
from src import config  # noqa: E402
from src.evaluation import (  # noqa: E402
    build_test_predictions,
    plot_clusters,
    plot_confusion,
    plot_residuals,
    write_json,
)
from src.modeling import BASELINE_CLF, BASELINE_REG, run_classification, run_clustering, run_regression  # noqa: E402
from src.preprocessing import clean_data, split_data  # noqa: E402
from src.validation import assert_no_leakage, check_expected_problems, validate_dataset  # noqa: E402


def run(student_id: str, seed: int, make_report: bool = True, verbose: bool = True) -> dict:
    """Recreate data, validate, train all tasks, save models and every artifact."""
    say = print if verbose else (lambda *a, **k: None)
    identity = config.parse_identity(student_id, seed)
    seed, code, digits = identity["seed"], identity["verification_code"], identity["reg_digits"]
    say(f"Verification code: {code}")
    say(f"Student id: {student_id} | seed: {seed}")

    csv_path = config.dataset_path(digits)
    write_csv(generate(seed), csv_path)
    fingerprint = sha256_file(csv_path)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(csv_path)
    base_rows = base_row_count(seed)
    quality = validate_dataset(raw)
    check_expected_problems(quality, base_rows)
    cleaned = clean_data(raw)
    (config.ARTIFACTS_DIR / "data_fingerprint.txt").write_text(
        f"verification_code: {code}\nfile: {csv_path.name}\nrows: {len(raw)}\nsha256: {fingerprint}\n",
        encoding="utf-8",
    )
    say(f"Dataset: {csv_path.name} | base rows {base_rows} | final rows {len(raw)}")
    say(f"Missing: soil_moisture_pct={quality['missing_by_column']['soil_moisture_pct']}, "
        f"fertilizer_kg={quality['missing_by_column']['fertilizer_kg']} | duplicates={quality['duplicate_rows']}")
    say(f"SHA-256 fingerprint: {fingerprint}")

    X_train, X_test, yr_train, yr_test, yc_train, yc_test = split_data(cleaned, seed)
    assert_no_leakage(X_train.columns)
    say(f"Split: train={len(X_train)} test={len(X_test)} (random_state={seed})")

    reg_results, reg_fitted, reg_sel = run_regression(X_train, yr_train, X_test, yr_test, seed)
    clf_results, clf_fitted, clf_sel = run_classification(X_train, yc_train, X_test, yc_test, seed)
    scores, best_k, cluster_model, profiles = run_clustering(X_train, seed)

    reg_model, clf_model = reg_fitted[reg_sel], clf_fitted[clf_sel]
    preds = build_test_predictions(X_test, yr_test, yc_test, reg_model, clf_model, cluster_model)
    preds.to_csv(config.ARTIFACTS_DIR / "test_predictions.csv", index=False, float_format="%.4f", lineterminator="\n")

    worst = preds.loc[preds["yield_abs_error"].idxmax()]
    wrong = preds[preds["actual_high_risk"] != preds["predicted_high_risk"]]
    reg_payload = {
        "verification_code": code,
        "target": config.TARGET_REG,
        "selection_rule": f"lowest {config.CV_FOLDS}-fold cross-validated RMSE on the training set (baseline excluded)",
        "baseline": BASELINE_REG,
        "models": reg_results,
        "selected_model": reg_sel,
        "selection_reason": (
            f"{reg_sel} had the lowest cross-validated RMSE on the training set "
            f"({reg_results[reg_sel]['cv_rmse_train']:.4f} tons) and beats the mean baseline "
            f"on the held-out test set (RMSE {reg_results[reg_sel]['test']['RMSE']:.4f} vs "
            f"{reg_results[BASELINE_REG]['test']['RMSE']:.4f})."
        ),
        "largest_error_case": {
            "row_id": int(worst["row_id"]),
            "actual_yield_tons": float(worst["actual_yield_tons"]),
            "predicted_yield_tons": float(worst["predicted_yield_tons"]),
            "error": float(worst["yield_error"]),
        },
    }
    clf_payload = {
        "verification_code": code,
        "target": config.TARGET_CLF,
        "positive_class": "high_risk = 1 (yield per hectare below the median)",
        "selection_rule": f"highest {config.CV_FOLDS}-fold cross-validated F1 on the training set (baseline excluded)",
        "baseline": BASELINE_CLF,
        "models": clf_results,
        "selected_model": clf_sel,
        "selection_reason": (
            f"{clf_sel} had the highest cross-validated F1 on the training set "
            f"({clf_results[clf_sel]['cv_f1_train']:.4f}); test F1 "
            f"{clf_results[clf_sel]['test']['f1']:.4f} vs {clf_results[BASELINE_CLF]['test']['f1']:.4f} for the baseline."
        ),
        "misclassified_test_rows": [
            {
                "row_id": int(r.row_id),
                "actual_high_risk": int(r.actual_high_risk),
                "predicted_high_risk": int(r.predicted_high_risk),
                "risk_probability": float(r.risk_probability),
            }
            for r in wrong.itertuples()
        ],
    }
    best_score = scores[best_k]
    cluster_payload = {
        "verification_code": code,
        "features_used": config.FEATURES,
        "targets_used": False,
        "silhouette_by_k": {str(k): v for k, v in scores.items()},
        "selected_k": int(best_k),
        "selection_reason": f"K={best_k} has the highest silhouette score ({best_score:.4f}).",
        "profiles": profiles,
        "interpretation": (
            "The silhouette scores are low, so the data has weak natural cluster structure; "
            "this is expected because the synthetic features were drawn independently. "
            "Clusters are useful as practical farm groupings, not as natural categories."
            if best_score < 0.25
            else "Clusters are reasonably separated."
        ),
    }
    write_json(config.ARTIFACTS_DIR / "regression_metrics.json", reg_payload)
    write_json(config.ARTIFACTS_DIR / "classification_metrics.json", clf_payload)
    write_json(config.ARTIFACTS_DIR / "clustering_report.json", cluster_payload)

    plot_residuals(preds, config.PLOTS_DIR / "residuals.png", code, reg_sel)
    plot_confusion(clf_results[clf_sel]["test"]["confusion_matrix"], config.PLOTS_DIR / "confusion_matrix.png", code, clf_sel)
    plot_clusters(cluster_model, pd.concat([X_train, X_test]), cluster_model.predict(pd.concat([X_train, X_test])),
                  config.PLOTS_DIR / "clusters.png", code, best_k)

    joblib.dump(reg_model.named_steps["prep"], config.MODELS_DIR / "preprocessor.joblib")
    joblib.dump(reg_model, config.MODELS_DIR / "regression_model.joblib")
    joblib.dump(clf_model, config.MODELS_DIR / "classification_model.joblib")
    joblib.dump(cluster_model, config.MODELS_DIR / "cluster_model.joblib")
    write_json(
        config.MODELS_DIR / "metadata.json",
        {
            "verification_code": code,
            "student_id": student_id,
            "seed": seed,
            "dataset_sha256": fingerprint,
            "features": config.FEATURES,
            "ranges": {k: list(v) for k, v in config.RANGES.items()},
            "regression_model": reg_sel,
            "classification_model": clf_sel,
            "selected_k": int(best_k),
            "cluster_profiles": profiles,
            "regression_test": reg_results[reg_sel]["test"],
            "classification_test": clf_results[clf_sel]["test"],
            "synthetic_data": True,
            "disclaimer": config.DISCLAIMER,
        },
    )

    say(f"Regression  selected: {reg_sel} | test MAE {reg_results[reg_sel]['test']['MAE']:.4f} "
        f"RMSE {reg_results[reg_sel]['test']['RMSE']:.4f} R2 {reg_results[reg_sel]['test']['R2']:.4f}")
    t = clf_results[clf_sel]["test"]
    say(f"Classifier  selected: {clf_sel} | precision {t['precision']:.4f} recall {t['recall']:.4f} "
        f"F1 {t['f1']:.4f} ROC-AUC {t['roc_auc']:.4f}")
    say(f"Clustering  selected K={best_k} (silhouette {best_score:.4f})")

    if make_report:
        try:
            from src.report import build_report
            pdf = build_report(identity, student_id)
            say(f"Report written: {pdf.relative_to(ROOT)}")
        except ImportError:
            say("reportlab not installed: technical report skipped (optional dependency).")

    from src.inference import load_bundle
    load_bundle.cache_clear()
    say("Artifacts:")
    for p in sorted(list(config.ARTIFACTS_DIR.rglob("*")) + list(config.MODELS_DIR.glob("*"))):
        if p.is_file():
            say(f"  {p.relative_to(ROOT)}")
    say(f"Done. Verification code {code}")
    return {"fingerprint": fingerprint, "code": code}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--no-report", action="store_true", help="Skip the PDF report.")
    args = parser.parse_args()
    try:
        run(args.student_id, args.seed, make_report=not args.no_report)
    except (config.IdentityError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
