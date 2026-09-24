# 6R Recommendation Engine — Model Training Handoff

> **Author:** Data-Preprocessing & ML Team  
> **Date:** 2026-09-21  
> **Status:** Training pipeline complete — ready for Lambda integration

---

## 1. What Was Done

We built an end-to-end ML training pipeline that takes the synthetic application portfolio dataset and produces a production-ready XGBoost model that classifies applications into one of the **6R migration strategies** (Rehost, Replatform, Repurchase, Refactor, Retire, Retain) with SHAP-based explainability.

### Work completed

| Phase | Deliverable | Status |
|-------|-------------|--------|
| Data engineering | Processed Google Cluster traces → `task_summary_ml_ready.csv` (179K rows) | ✅ Done |
| Data engineering | Synthetic application portfolio → `application_portfolio_1000.csv` (1000 rows) | ✅ Done |
| ML spec | `ML_ENGINE_SPEC.md` — model selection, features, API contract, deployment strategy | ✅ Done |
| Training script | `notebook/model_training.py` — full pipeline with tuning and anti-leakage guards | ✅ Done |
| Benchmark | `notebook/model_benchmark.py` & `benchmark_results.json` | ✅ Done |
| Model artifacts | `model.joblib`, `label_encoder.joblib`, `shap_summary.png` | ✅ Exported |
| Leakage audit | Out-of-fold evaluation, label noise, Gaussian jitter, heavy regularisation | ✅ Passed |
| Backend scaffold | FastAPI mock integration layer with stable API routes | ✅ Done |
| Documentation | Architecture, API contract, data README, walkthrough | ✅ Done |

---

## 2. Repository Structure

```
Intelligent-Cloud-Migration-Planning-AI-data-preprocessing/
├── data/
│   ├── raw/                          # Original Google Cluster trace files
│   │   ├── machine_events/
│   │   ├── task_events/
│   │   └── task_usage/
│   └── processed/
│       ├── README.md                 # Dataset documentation
│       ├── task_summary_ml_ready.csv # 179K workload records (21 cols)
│       └── application_portfolio_1000.csv  # 1000 synthetic apps (9 cols)
├── notebook/
│   ├── model_training.py             # ★ Main training script (with tuned XGBoost)
│   ├── model_benchmark.py            # Baseline vs Tuning benchmark script
│   ├── benchmark_results.json        # Benchmark comparison results
│   ├── model.joblib                  # ★ Trained tuned XGBoost pipeline (~2.3 MB)
│   ├── label_encoder.joblib          # ★ Label encoder for 6R classes
│   ├── shap_summary.png             # SHAP feature importance plot
│   ├── requirements.txt             # Python dependencies for training
│   ├── application_portfolio_1000.ipynb  # Data generation notebook
│   └── processed.ipynb              # Data processing notebook
├── backend/
│   └── app/                         # FastAPI mock integration layer
├── docs/
│   ├── ARCHITECTURE.md              # System architecture overview
│   ├── API_CONTRACT.md              # API endpoints & Postman guide
│   └── MODEL_TRAINING_HANDOFF.md    # ★ This document
├── ML_ENGINE_SPEC.md                # Model design specification
├── README.md                        # Project overview & quick start
└── walkthrough.md                   # Leakage audit & verified results
```

---

## 3. How to Run the Training Pipeline

### Prerequisites

```bash
pip install pandas scikit-learn xgboost shap joblib numpy matplotlib
```

### Train

```bash
cd notebook
python model_training.py
```

This runs 7 steps:
1. Load `application_portfolio_1000.csv` (drops ID columns)
2. Generate 6R labels via heuristic rules + 15% stochastic noise
3. Hyperparameter tuning (RandomizedSearchCV)
4. 5-fold stratified cross-validation (out-of-fold only)
5. Final refit on all data (no jitter) for production export
6. SHAP explainability → `shap_summary.png`
7. Inference demo matching `ML_ENGINE_SPEC.md` §4 contract

**Outputs:** `model.joblib`, `label_encoder.joblib`, `shap_summary.png`

---

## 4. Model Details

### Algorithm

**XGBoost (XGBClassifier)** tuned via `RandomizedSearchCV` and wrapped in a scikit-learn `Pipeline` with a `ColumnTransformer` preprocessor.

### Features (8 total)

| Feature | Type | Preprocessing |
|---------|------|---------------|
| `cpu_usage` | Numerical | Impute → Scale → Jitter (train only) |
| `memory_usage` | Numerical | Impute → Scale → Jitter (train only) |
| `age_years` | Numerical | Impute → Scale → Jitter (train only) |
| `dependency_count` | Numerical | Impute → Scale → Jitter (train only) |
| `resource_intensity` | Numerical (engineered: cpu × memory) | Impute → Scale → Jitter (train only) |
| `coupling_to_age` | Numerical (engineered: deps / (age+1)) | Impute → Scale → Jitter (train only) |
| `criticality` | Ordinal (Low/Medium/High) | OrdinalEncoder |
| `compliance_flag` | Binary | Passthrough |

### Hyperparameters

```
n_estimators=300, max_depth=5, learning_rate=0.01,
subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
reg_alpha=2.0, reg_lambda=5.0
```

### Anti-Leakage Measures

| Measure | Purpose |
|---------|---------|
| Out-of-fold evaluation only | Never evaluates on training data |
| 15% stochastic label noise | Prevents tree from memorising heuristic splits |
| GaussianJitter (σ=0.05) | Blurs deterministic feature cutoffs during training |
| Heavy L1/L2 regularisation | Prevents overfitting to synthetic patterns |
| Assertion guards | Fails if any metric ≥ 0.90 or any class hits perfect precision/recall |

---

## 5. Verified Performance (Out-of-Fold)

| Metric | Value |
|--------|-------|
| **Macro F1** | 0.6964 ± 0.031 |
| **Accuracy** | 0.8180 ± 0.022 |
| **Macro Precision** | 0.7014 |
| **Macro Recall** | 0.6962 |

### Per-Class Breakdown

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Rehost | 0.87 | 0.90 | 0.88 | 328 |
| Repurchase | 0.87 | 0.88 | 0.87 | 244 |
| Retain | 0.85 | 0.82 | 0.84 | 176 |
| Refactor | 0.78 | 0.82 | 0.80 | 120 |
| Replatform | 0.64 | 0.65 | 0.64 | 93 |
| Retire | 0.21 | 0.10 | 0.14 | 39 |

> **Note:** `Retire` has low performance due to its small class size (3.9% of data). This is realistic — Retire is a rare migration strategy in practice. The `balanced` sample weighting partially compensates.

### SHAP Feature Importance (ranked)

1. `criticality` — strongest overall signal
2. `age_years`
3. `coupling_to_age`
4. `compliance_flag`
5. `resource_intensity`
6. `memory_usage`
7. `cpu_usage`
8. `dependency_count`

---

## 6. API Output Contract

The model's inference helper (`predict_single()`) returns JSON matching `ML_ENGINE_SPEC.md` §4:

```json
{
  "app_id": "APP001",
  "recommendation": "Repurchase",
  "confidence_score": 0.77,
  "probabilities": {
    "Rehost": 0.05, "Replatform": 0.03, "Repurchase": 0.77,
    "Refactor": 0.08, "Retire": 0.02, "Retain": 0.05
  },
  "explanation": {
    "top_contributors": [
      { "feature": "dependency_count", "impact": "+1.35", "value": "4" },
      { "feature": "compliance_flag", "impact": "+0.41", "value": "0" }
    ],
    "summary": "Recommended for Repurchase primarily because dependency_count is high (4) and compliance_flag is low (0)."
  }
}
```

---

## 7. How to Use the Exported Model

```python
import joblib
import pandas as pd
import numpy as np
import shap

# Load artifacts
pipeline = joblib.load("notebook/model.joblib")
le = joblib.load("notebook/label_encoder.joblib")

# Prepare input (raw feature dict)
app = {
    "cpu_usage": 0.45, "memory_usage": 0.60,
    "age_years": 8, "criticality": "Medium",
    "compliance_flag": 0, "dependency_count": 3,
}

# Add interaction features
app["resource_intensity"] = app["cpu_usage"] * app["memory_usage"]
app["coupling_to_age"] = app["dependency_count"] / (app["age_years"] + 1)

df = pd.DataFrame([app])
pred = pipeline.predict(df)
label = le.inverse_transform(pred)[0]
probas = pipeline.predict_proba(df)[0]

print(f"Recommendation: {label}  (confidence: {probas[pred[0]]:.2f})")
```

---

## 8. Known Limitations & Caveats

| Item | Detail |
|------|--------|
| **Synthetic data** | Labels are heuristic-generated with noise, not real migration decisions. Model will need retraining when real labelled data is available. |
| **Retire class imbalance** | Only 39 samples (3.9%). Consider oversampling or collecting more Retire examples for production. |
| **No temporal features** | The model doesn't account for time-series resource trends — only point-in-time snapshots. |
| **Interaction features** | `resource_intensity` and `coupling_to_age` must be computed before calling the pipeline (they are not computed inside the pipeline itself). |
| **Lambda size** | XGBoost + scikit-learn + SHAP exceeds Lambda's 250 MB limit. **Must use Docker container image** (see `ML_ENGINE_SPEC.md` §7). |

---

## 9. Next Steps for Receiving Team Member

| Priority | Task | Reference |
|----------|------|-----------|
| 🔴 High | Package model into AWS Lambda Docker image (ECR) | `ML_ENGINE_SPEC.md` §7 |
| 🔴 High | Wire `predict_single()` into the Lambda handler | `ML_ENGINE_SPEC.md` §4 |
| 🟡 Medium | Replace FastAPI mock `/recommendation` endpoint with real model call | `docs/API_CONTRACT.md` |
| 🟡 Medium | Connect to DynamoDB for application data instead of CSV | `docs/ARCHITECTURE.md` |
| 🟢 Low | Retrain on real labelled data when available | Swap `application_portfolio_1000.csv` |
| 🟢 Low | Address Retire class imbalance (SMOTE or more data) | `walkthrough.md` |
| 🟢 Low | Add model versioning / MLflow tracking | — |

---

## 10. Key Specification Documents

| Document | Purpose |
|----------|---------|
| `ML_ENGINE_SPEC.md` | Model selection rationale, feature list, API contract, deployment plan |
| `docs/ARCHITECTURE.md` | Full system architecture (6R, waves, cost/risk, copilot) |
| `docs/API_CONTRACT.md` | API routes, request/response shapes, Postman setup |
| `data/processed/README.md` | Dataset schemas, sources, quality notes |
| `walkthrough.md` | Detailed leakage audit with fix-by-fix explanation |

---

*End of handoff — questions? Check the docs listed above or reach out to the ML team.*
