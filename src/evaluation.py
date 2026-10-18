"""Writing metrics, prediction tables and plots from fitted models."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def build_test_predictions(X_test, y_reg_test, y_clf_test, reg_model, clf_model, cluster_model) -> pd.DataFrame:
    pred = reg_model.predict(X_test)
    proba = clf_model.predict_proba(X_test)[:, 1]
    return pd.DataFrame(
        {
            "row_id": X_test.index.to_numpy(),
            "actual_yield_tons": y_reg_test.to_numpy(),
            "predicted_yield_tons": pred,
            "yield_error": pred - y_reg_test.to_numpy(),
            "yield_abs_error": np.abs(pred - y_reg_test.to_numpy()),
            "actual_high_risk": y_clf_test.to_numpy(),
            "predicted_high_risk": clf_model.predict(X_test),
            "risk_probability": proba,
            "cluster": cluster_model.predict(X_test),
        }
    )


def plot_residuals(df: pd.DataFrame, path: Path, code: str, model_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(df["predicted_yield_tons"], df["yield_error"], alpha=0.75)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xlabel("Predicted yield (tons)")
    axes[0].set_ylabel("Error (predicted - actual)")
    axes[0].set_title("Residuals vs prediction")
    axes[1].hist(df["yield_error"], bins=15, edgecolor="black")
    axes[1].set_xlabel("Error (tons)")
    axes[1].set_ylabel("Farms")
    axes[1].set_title("Residual distribution")
    fig.suptitle(f"{model_name} on the test set  |  {code}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(cm, path: Path, code: str, model_name: str) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Not high risk", "High risk"])
    ax.set_yticks([0, 1], ["Not high risk", "High risk"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_title(f"{model_name}\n{code}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_clusters(cluster_model, X_all, labels, path: Path, code: str, k: int) -> None:
    Z = cluster_model.named_steps["prep"].transform(X_all)
    pca = PCA(n_components=2, random_state=0)
    P = pca.fit_transform(Z)
    centers = pca.transform(cluster_model.named_steps["kmeans"].cluster_centers_)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    scatter = ax.scatter(P[:, 0], P[:, 1], c=labels, cmap="tab10", alpha=0.75, s=28)
    ax.scatter(centers[:, 0], centers[:, 1], c="black", marker="X", s=130, label="Centers")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} variance)")
    ax.set_title(f"Farm clusters (K={k})  |  {code}")
    ax.legend(*scatter.legend_elements(), title="Cluster", loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
