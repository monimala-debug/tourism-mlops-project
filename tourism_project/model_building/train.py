"""
train.py - Tune, evaluate and save the purchase-propensity model, with MLflow tracking.

Steps
1. Load the four split files written by prep.py.
2. Build ONE sklearn Pipeline: preprocessing (imputation + one-hot encoding) + Random Forest.
3. Tune a small, transparent grid with 5-fold stratified cross-validation, selecting by
   average precision (area under the precision-recall curve).
4. Log every candidate (nested MLflow run) plus the chosen parameters and test metrics.
   MLflow uses a local SQLite file - no server, no ngrok, nothing exposed publicly.
5. Evaluate once on the untouched test set.
6. Save the fitted pipeline to tourism_project/deployment/model.joblib for the app.

Run from the repository root:
    python tourism_project/model_building/train.py
"""

import argparse
import json
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # silence an informational MLflow banner
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend for CI / Colab
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score,
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "ProdTaken"
RANDOM_STATE = 42
CV_FOLDS = 5
DECISION_THRESHOLD = 0.5  # default cut-off used for the yes/no metrics and in the app
EXPERIMENT_NAME = "tourism-purchase-propensity"

# Categorical columns are identified by name; everything else is treated as numeric.
CATEGORICAL_FEATURES = ["Occupation", "Gender", "MaritalStatus", "Designation"]

# Modest grid: 2 x 3 x 2 x 2 = 24 candidates x 5 folds = 120 fits.
PARAM_GRID = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [6, 12, None],
    "model__min_samples_leaf": [1, 5],
    "model__max_features": ["sqrt", 0.5],
}
# Scores recorded for every candidate; "average_precision" picks the winner.
SCORING = {
    "average_precision": "average_precision",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall",
}
SELECTION_METRIC = "average_precision"


def build_pipeline(numeric_features, categorical_features) -> Pipeline:
    numeric_steps = SimpleImputer(strategy="median")
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # Categories seen fewer than 10 times in a training fold (e.g. "Freelancer")
        # are grouped into one "infrequent" column; unseen labels at prediction time are
        # mapped to that same column instead of raising an error.
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=10, sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_steps, numeric_features),
        ("cat", categorical_steps, categorical_features),
    ])
    model = RandomForestClassifier(
        class_weight="balanced_subsample",  # up-weights the minority (buyer) class
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def test_metrics(y_true, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    metrics = {
        "test_accuracy": accuracy_score(y_true, pred),
        "test_precision": precision_score(y_true, pred, zero_division=0),
        "test_recall": recall_score(y_true, pred, zero_division=0),
        "test_f1": f1_score(y_true, pred, zero_division=0),
        "test_positive_rate_baseline": float(np.mean(y_true)),
    }
    # AUC metrics are only defined when both classes are present in y_true.
    if len(np.unique(y_true)) == 2:
        metrics["test_roc_auc"] = roc_auc_score(y_true, proba)
        metrics["test_pr_auc_average_precision"] = average_precision_score(y_true, proba)
    return {k: float(v) for k, v in metrics.items()}


def top_k_table(y_true, proba, fractions=(0.10, 0.20, 0.30)) -> pd.DataFrame:
    """Marketing view: if we only call the top X% highest-scored customers, how many
    of them are buyers (precision) and what share of all buyers do we reach (recall)?"""
    order = np.argsort(-proba)
    y_sorted = np.asarray(y_true)[order]
    rows = []
    for frac in fractions:
        k = max(1, int(round(frac * len(y_sorted))))
        hits = int(y_sorted[:k].sum())
        rows.append({
            "contact_top": f"{frac:.0%}",
            "customers_contacted": k,
            "buyers_found": hits,
            "precision_in_group": round(hits / k, 3),
            "share_of_all_buyers": round(hits / max(1, int(y_sorted.sum())), 3),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and track the tourism model")
    parser.add_argument("--splits-dir", type=Path, default=Path("tourism_project/data/splits"))
    parser.add_argument("--model-out", type=Path,
                        default=Path("tourism_project/deployment/model.joblib"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("tourism_project/outputs/training"))
    args = parser.parse_args()

    # ---------------------------------------------------------------- load splits
    X_train = pd.read_csv(args.splits_dir / "Xtrain.csv")
    X_test = pd.read_csv(args.splits_dir / "Xtest.csv")
    y_train = pd.read_csv(args.splits_dir / "ytrain.csv")[TARGET].astype(int)
    y_test = pd.read_csv(args.splits_dir / "ytest.csv")[TARGET].astype(int)

    if list(X_train.columns) != list(X_test.columns):
        raise SystemExit("Xtrain and Xtest have different columns - rerun prep.py")
    categorical = [c for c in X_train.columns if c in CATEGORICAL_FEATURES]
    numeric = [c for c in X_train.columns if c not in CATEGORICAL_FEATURES]
    X_train[numeric] = X_train[numeric].astype("float64")
    X_test[numeric] = X_test[numeric].astype("float64")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Numeric features: {numeric}\nCategorical features: {categorical}")

    # ---------------------------------------------------------------- MLflow (local)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir.resolve()
    mlflow.set_tracking_uri(f"sqlite:///{out / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=(out / "mlartifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ---------------------------------------------------------------- tuning
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=build_pipeline(numeric, categorical),
        param_grid=PARAM_GRID,
        scoring=SCORING,
        refit=SELECTION_METRIC,
        cv=cv,
        n_jobs=1,  # the forest itself already uses all cores
        return_train_score=False,
    )

    run_name = "rf-grid-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    with mlflow.start_run(run_name=run_name) as parent:
        mlflow.log_params({
            "algorithm": "RandomForestClassifier",
            "class_weight": "balanced_subsample",
            "cv_folds": CV_FOLDS,
            "selection_metric": SELECTION_METRIC,
            "random_state": RANDOM_STATE,
            "decision_threshold": DECISION_THRESHOLD,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "features": ",".join(X_train.columns),
            "sklearn_version": sklearn.__version__,
        })

        n_candidates = int(np.prod([len(v) for v in PARAM_GRID.values()]))
        print(f"\nGrid search: {n_candidates} candidates x {CV_FOLDS} folds")
        search.fit(X_train, y_train)

        # One nested run per attempted parameter combination.
        cv_results = pd.DataFrame(search.cv_results_)
        for i, row in cv_results.iterrows():
            with mlflow.start_run(run_name=f"candidate-{i:02d}", nested=True):
                mlflow.log_params({k.replace("model__", ""): str(v)
                                   for k, v in row["params"].items()})
                mlflow.log_metrics({
                    f"cv_{m}_mean": float(row[f"mean_test_{m}"]) for m in SCORING
                } | {
                    f"cv_{m}_std": float(row[f"std_test_{m}"]) for m in SCORING
                } | {"rank_by_selection_metric": int(row[f"rank_test_{SELECTION_METRIC}"])})

        best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        mlflow.log_params({f"best_{k}": str(v) for k, v in best_params.items()})
        mlflow.log_metric(f"best_cv_{SELECTION_METRIC}", float(search.best_score_))

        # ------------------------------------------------------------ single test evaluation
        best_model = search.best_estimator_
        proba = best_model.predict_proba(X_test)[:, 1]
        pred = (proba >= DECISION_THRESHOLD).astype(int)
        metrics = test_metrics(y_test, proba, DECISION_THRESHOLD)
        mlflow.log_metrics(metrics)

        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        report_text = classification_report(y_test, pred, digits=3, zero_division=0)
        topk = top_k_table(y_test, proba)

        # ------------------------------------------------------------ save artifacts
        summary_cols = ["params"] + [f"mean_test_{m}" for m in SCORING] + [f"rank_test_{SELECTION_METRIC}"]
        cv_table = cv_results[summary_cols].sort_values(f"rank_test_{SELECTION_METRIC}")
        cv_table.to_csv(out / "cv_results.csv", index=False)
        (out / "classification_report.txt").write_text(report_text)
        topk.to_csv(out / "top_k_outreach.csv", index=False)

        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["No purchase (0)", "Purchase (1)"]).plot(
            ax=ax, colorbar=False, values_format="d")
        ax.set_title(f"Test confusion matrix (threshold {DECISION_THRESHOLD})")
        fig.tight_layout()
        fig.savefig(out / "confusion_matrix.png", dpi=120)
        plt.close(fig)

        results = {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mlflow_run_id": parent.info.run_id,
            "best_params": {k: (v if v is None or isinstance(v, (int, float, str)) else str(v))
                            for k, v in best_params.items()},
            f"best_cv_{SELECTION_METRIC}": float(search.best_score_),
            "test_metrics": metrics,
            "confusion_matrix": {"labels": [0, 1], "matrix": cm.tolist()},
            "features": list(X_train.columns),
            "numeric_features": numeric,
            "categorical_features": categorical,
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                         "pandas": pd.__version__, "numpy": np.__version__},
        }
        (out / "metrics.json").write_text(json.dumps(results, indent=2))

        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, args.model_out, compress=3)

        for name in ("metrics.json", "cv_results.csv", "classification_report.txt",
                     "top_k_outreach.csv", "confusion_matrix.png"):
            mlflow.log_artifact(str(out / name))
        mlflow.log_artifact(str(args.model_out))

    # ---------------------------------------------------------------- printed summary
    lines = [
        "## Training summary",
        f"- MLflow experiment: `{EXPERIMENT_NAME}`, parent run `{parent.info.run_id}` "
        f"({n_candidates} nested candidate runs)",
        f"- Best CV {SELECTION_METRIC}: **{search.best_score_:.4f}**",
        f"- Best parameters: `{best_params}`",
        "",
        "| Test metric | Value |",
        "|---|---|",
        *[f"| {k} | {v:.4f} |" for k, v in metrics.items()],
        "",
        f"Confusion matrix (rows = actual 0/1, cols = predicted 0/1): `{cm.tolist()}`",
        "",
        "Top 5 candidates by CV average precision:",
        "",
        cv_table.head(5).to_markdown(index=False) if _has_tabulate() else cv_table.head(5).to_string(index=False),
        "",
        "Outreach view on the test set (contact only the highest-scored customers):",
        "",
        topk.to_markdown(index=False) if _has_tabulate() else topk.to_string(index=False),
        "",
        f"Model saved to `{args.model_out}` "
        f"({args.model_out.stat().st_size / 1e6:.2f} MB)",
    ]
    summary_md = "\n".join(lines)
    (out / "training_summary.md").write_text(summary_md)
    print("\n" + summary_md)
    print("\nClassification report (test set):\n" + report_text)

    # In GitHub Actions, also show the summary on the run's summary page.
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as fh:
            fh.write(summary_md + "\n")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401  (optional, only used for nicer tables)
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
