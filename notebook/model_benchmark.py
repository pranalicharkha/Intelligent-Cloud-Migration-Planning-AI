"""
ML Model Benchmark — Baseline Comparison + Hyperparameter Tuning
=================================================================
Trains 4 models (Random Forest, XGBoost, LightGBM, Logistic Regression)
with default params, then tunes each via RandomizedSearchCV, compares
default vs tuned performance, and exports the best model.

Reuses data pipeline from model_training.py. Does NOT modify existing
model.joblib or model_training.py.

Self-verifying: run `python model_benchmark.py` — prints comparison tables.
"""

import json
import pathlib
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Reuse existing data pipeline
from model_training import (
    CLASSES_6R,
    NUM_FEATURES,
    ORD_FEATURES,
    BIN_FEATURES,
    load_data,
    generate_labels,
    add_interaction_features,
    build_preprocessor,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

OUT_DIR = pathlib.Path(__file__).resolve().parent
ALL_FEATURES = NUM_FEATURES + ORD_FEATURES + BIN_FEATURES
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ---------------------------------------------------------------------------
# §1 — Model definitions (default params)
# ---------------------------------------------------------------------------

def _default_models() -> dict[str, any]:
    """Return models with reasonable default hyperparameters."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=2.0,
            reg_lambda=3.0,
            eval_metric="mlogloss",
            random_state=42,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=2.0,
            reg_lambda=3.0,
            random_state=42,
            verbose=-1,
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="saga",
            max_iter=1000,
            random_state=42,
        ),
    }


# ---------------------------------------------------------------------------
# §2 — Tuning search spaces
# ---------------------------------------------------------------------------

PARAM_GRIDS = {
    "Random Forest": {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [3, 5, 7, 10, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", 0.5],
    },
    "XGBoost": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [2, 3, 4, 5],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.6, 0.7, 0.8],
        "classifier__colsample_bytree": [0.6, 0.7, 0.8],
        "classifier__reg_alpha": [0.5, 1.0, 2.0],
        "classifier__reg_lambda": [1.0, 3.0, 5.0],
        "classifier__min_child_weight": [3, 5, 7],
    },
    "LightGBM": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__num_leaves": [15, 31, 50],
        "classifier__max_depth": [3, 5, 7, -1],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.6, 0.7, 0.8],
        "classifier__colsample_bytree": [0.6, 0.7, 0.8],
        "classifier__reg_alpha": [0.0, 0.5, 1.0],
        "classifier__reg_lambda": [0.0, 1.0, 3.0],
    },
    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "classifier__penalty": ["l1", "l2"],
        "classifier__solver": ["saga"],
        "classifier__max_iter": [1000],
    },
}


# ---------------------------------------------------------------------------
# §3 — Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_oof(model_name: str, pipeline: Pipeline, X: pd.DataFrame,
                  y: np.ndarray, le: LabelEncoder) -> dict:
    """Out-of-fold evaluation. Returns metrics dict."""
    oof_preds = np.empty_like(y)
    f1_scores, acc_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(CV.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        sw = compute_sample_weight("balanced", y_tr)
        pipe = _clone_pipeline(pipeline)

        # Pass sample weights — different param name per classifier
        clf_name = type(pipe.named_steps["classifier"]).__name__
        if clf_name == "LogisticRegression":
            pipe.fit(X_tr, y_tr, classifier__sample_weight=sw)
        else:
            pipe.fit(X_tr, y_tr, classifier__sample_weight=sw)

        preds = pipe.predict(X_val)
        oof_preds[val_idx] = preds
        f1_scores.append(f1_score(y_val, preds, average="macro"))
        acc_scores.append(accuracy_score(y_val, preds))

    oof_f1 = f1_score(y, oof_preds, average="macro")
    oof_acc = accuracy_score(y, oof_preds)
    oof_prec = precision_score(y, oof_preds, average="macro")
    oof_rec = recall_score(y, oof_preds, average="macro")

    per_class = classification_report(y, oof_preds, target_names=le.classes_,
                                      output_dict=True)

    return {
        "model": model_name,
        "macro_f1": round(oof_f1, 4),
        "accuracy": round(oof_acc, 4),
        "macro_precision": round(oof_prec, 4),
        "macro_recall": round(oof_rec, 4),
        "cv_f1_mean": round(np.mean(f1_scores), 4),
        "cv_f1_std": round(np.std(f1_scores), 4),
        "cv_acc_mean": round(np.mean(acc_scores), 4),
        "cv_acc_std": round(np.std(acc_scores), 4),
        "per_class": {
            cls: {k: round(v, 4) for k, v in per_class[cls].items()}
            for cls in le.classes_
        },
    }


def _clone_pipeline(pipeline: Pipeline) -> Pipeline:
    """Clone a pipeline (fresh unfitted copy)."""
    from sklearn.base import clone
    return clone(pipeline)


def _build_pipeline(classifier, jitter: bool = True) -> Pipeline:
    """Wrap a classifier with the standard preprocessor."""
    return Pipeline([
        ("preprocessor", build_preprocessor(jitter=jitter)),
        ("classifier", classifier),
    ])


# ---------------------------------------------------------------------------
# §4 — Phase 1: Default params comparison
# ---------------------------------------------------------------------------

def run_defaults(X: pd.DataFrame, y: np.ndarray, le: LabelEncoder) -> list[dict]:
    """Evaluate all models with default hyperparameters."""
    print("=" * 65)
    print("PHASE 1: Default-Params Baseline Comparison")
    print("=" * 65)

    results = []
    models = _default_models()

    for name, clf in models.items():
        print(f"\n  [{name}] Running 5-fold CV ...")
        jitter = name != "Logistic Regression"
        pipeline = _build_pipeline(clf, jitter=jitter)
        metrics = _evaluate_oof(name, pipeline, X, y, le)
        results.append(metrics)
        print(f"    macro-F1: {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}"
              f"   accuracy: {metrics['cv_acc_mean']:.4f} ± {metrics['cv_acc_std']:.4f}")

    _print_comparison("Default Params", results)
    return results


# ---------------------------------------------------------------------------
# §5 — Phase 2: Hyperparameter tuning
# ---------------------------------------------------------------------------

def run_tuning(X: pd.DataFrame, y: np.ndarray, le: LabelEncoder) -> list[dict]:
    """Tune all models with RandomizedSearchCV."""
    print("\n" + "=" * 65)
    print("PHASE 2: Hyperparameter Tuning (RandomizedSearchCV)")
    print("=" * 65)

    results = []
    best_params_all = {}
    models = _default_models()

    sw = compute_sample_weight("balanced", y)

    for name, clf in models.items():
        print(f"\n  [{name}] Tuning (30 iterations × 5 folds) ...")
        t0 = time.time()

        jitter = name != "Logistic Regression"
        pipeline = _build_pipeline(clf, jitter=jitter)

        n_iter = min(30, _grid_size(PARAM_GRIDS[name]))

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=PARAM_GRIDS[name],
            n_iter=n_iter,
            scoring="f1_macro",
            cv=CV,
            random_state=42,
            n_jobs=-1,
            error_score="raise",
        )

        # Fit with sample weights
        fit_params = {"classifier__sample_weight": sw}
        search.fit(X, y, **fit_params)

        elapsed = time.time() - t0
        print(f"    Best CV F1: {search.best_score_:.4f}  ({elapsed:.1f}s)")

        # Strip pipeline prefix from params for readability
        clean_params = {
            k.replace("classifier__", ""): v
            for k, v in search.best_params_.items()
        }
        best_params_all[name] = clean_params
        print(f"    Best params: {clean_params}")

        # Now evaluate OOF with best estimator for full metrics
        metrics = _evaluate_oof(name, search.best_estimator_, X, y, le)
        metrics["best_params"] = clean_params
        metrics["tuning_time_s"] = round(elapsed, 1)
        results.append(metrics)

    _print_comparison("Tuned Params", results)
    return results


def _grid_size(grid: dict) -> int:
    """Approximate number of unique combinations in the grid."""
    size = 1
    for v in grid.values():
        size *= len(v)
    return size


# ---------------------------------------------------------------------------
# §6 — Comparison display
# ---------------------------------------------------------------------------

def _print_comparison(title: str, results: list[dict]):
    """Print a formatted comparison table."""
    print(f"\n{'-' * 65}")
    print(f"  {title} - Comparison Table")
    print(f"{'-' * 65}")
    print(f"  {'Model':<22} {'Macro-F1':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}")
    print(f"  {'-' * 62}")
    for r in sorted(results, key=lambda x: x["macro_f1"], reverse=True):
        print(f"  {r['model']:<22} {r['macro_f1']:>10.4f} {r['accuracy']:>10.4f}"
              f" {r['macro_precision']:>10.4f} {r['macro_recall']:>10.4f}")
    print()


def _print_default_vs_tuned(default_results: list[dict], tuned_results: list[dict]):
    """Print default vs tuned improvement table."""
    print(f"\n{'=' * 65}")
    print("PHASE 3: Default vs Tuned Comparison")
    print(f"{'=' * 65}")

    default_map = {r["model"]: r for r in default_results}
    tuned_map = {r["model"]: r for r in tuned_results}

    print(f"\n  {'Model':<22} {'Default F1':>11} {'Tuned F1':>10} {'Delta F1':>10} {'Improved?':>10}")
    print(f"  {'-' * 62}")

    for name in default_map:
        d_f1 = default_map[name]["macro_f1"]
        t_f1 = tuned_map[name]["macro_f1"]
        delta = t_f1 - d_f1
        improved = "+ Yes" if delta > 0.001 else ("~ Same" if abs(delta) <= 0.001 else "- No")
        print(f"  {name:<22} {d_f1:>11.4f} {t_f1:>10.4f} {delta:>+8.4f} {improved:>10}")

    # Overall best
    best_tuned = max(tuned_results, key=lambda x: x["macro_f1"])
    print(f"\n  * Best model: {best_tuned['model']} "
          f"(tuned macro-F1: {best_tuned['macro_f1']:.4f})")
    if "best_params" in best_tuned:
        print(f"    Best params: {best_tuned['best_params']}")

    # Per-class breakdown for best model
    print(f"\n  Per-class breakdown ({best_tuned['model']}):")
    print(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-' * 55}")
    for cls, metrics in best_tuned["per_class"].items():
        print(f"  {cls:<15} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f}"
              f" {metrics['f1-score']:>10.4f} {metrics['support']:>10.0f}")

    return best_tuned


# ---------------------------------------------------------------------------
# §7 — Export best model
# ---------------------------------------------------------------------------

def export_best(best_result: dict, X: pd.DataFrame, y: np.ndarray,
                le: LabelEncoder):
    """Retrain the best model on all data (no jitter) and export."""
    print(f"\n{'-' * 65}")
    print(f"  Exporting best model: {best_result['model']}")
    print(f"{'-' * 65}")

    models = _default_models()
    clf = models[best_result["model"]]

    # Apply tuned params
    if "best_params" in best_result:
        clf.set_params(**best_result["best_params"])

    pipeline = _build_pipeline(clf, jitter=False)
    sw = compute_sample_weight("balanced", y)
    pipeline.fit(X, y, classifier__sample_weight=sw)

    # Verify it can predict
    sample = X.iloc[:1]
    pred = pipeline.predict(sample)
    label = le.inverse_transform(pred)[0]
    print(f"  Sanity check: sample prediction = {label} [OK]")

    model_path = OUT_DIR / "best_model.joblib"
    le_path = OUT_DIR / "best_label_encoder.joblib"
    joblib.dump(pipeline, model_path)
    joblib.dump(le, le_path)
    print(f"  Exported: {model_path}")
    print(f"  Exported: {le_path}")

    return model_path, le_path


# ---------------------------------------------------------------------------
# §8 — Leakage assertions
# ---------------------------------------------------------------------------

def _assert_no_leakage(results: list[dict]):
    """Fail if any model shows suspiciously high metrics."""
    for r in results:
        name = r["model"]
        assert r["macro_f1"] < 0.90, f"LEAKAGE: {name} macro-F1 {r['macro_f1']:.4f} >= 0.90"
        assert r["accuracy"] < 0.95, f"LEAKAGE: {name} accuracy {r['accuracy']:.4f} >= 0.95"
        for cls, m in r["per_class"].items():
            assert m["precision"] < 1.0, f"LEAKAGE: {name}/{cls} perfect precision"
            assert m["recall"] < 1.0, f"LEAKAGE: {name}/{cls} perfect recall"
    print("  [OK] No leakage detected across all models")


# ---------------------------------------------------------------------------
# §9 — Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("ML Model Benchmark — 6R Recommendation Engine")
    print("=" * 65)

    # Data pipeline (reused from model_training.py)
    print("\n[Data] Loading and preparing ...")
    df = load_data()
    labels = generate_labels(df)
    df = add_interaction_features(df)

    le = LabelEncoder()
    le.fit(CLASSES_6R)
    y = le.transform(labels)

    X = df[ALL_FEATURES]
    print(f"  {len(X)} samples, {len(ALL_FEATURES)} features, {len(CLASSES_6R)} classes")
    print(f"  Class distribution: {dict(zip(*np.unique(le.inverse_transform(y), return_counts=True)))}")

    # Phase 1: Defaults
    default_results = run_defaults(X, y, le)
    _assert_no_leakage(default_results)

    # Phase 2: Tuning
    tuned_results = run_tuning(X, y, le)
    _assert_no_leakage(tuned_results)

    # Phase 3: Comparison + export
    best = _print_default_vs_tuned(default_results, tuned_results)
    model_path, le_path = export_best(best, X, y, le)

    # Export results JSON
    all_results = {
        "default": default_results,
        "tuned": tuned_results,
        "best_model": best["model"],
        "best_macro_f1": best["macro_f1"],
    }
    results_path = OUT_DIR / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results exported: {results_path}")

    print("\n" + "=" * 65)
    print("Benchmark complete.")
    print(f"  Best model : {best['model']} (macro-F1: {best['macro_f1']:.4f})")
    print(f"  Artifacts  : {model_path.name}, {le_path.name}, {results_path.name}")
    print("=" * 65)


if __name__ == "__main__":
    main()
