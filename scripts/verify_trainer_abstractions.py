"""
Verification script for Week 1 Task 1.4:
Core model training abstractions, interfaces, and cross-validation contracts.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.trainer import (
    TrainingConfig,
    EvaluationMetrics,
    PatientGroupKFoldSplitter,
    ClinicalMetricsEvaluator,
    SklearnEstimatorWrapper,
    XGBoostEstimatorWrapper,
    LightGBMEstimatorWrapper,
    RespiratoryPredictorSuite,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def verify_abstractions():
    print("=" * 65)
    print("MIMIC-IV Model Training Abstractions Verification (Task 1.4)")
    print("=" * 65)

    # 1. Configuration & Metrics Schemas
    print("\n[*] 1. Testing Configuration and Metric Schemas...")
    cfg = TrainingConfig()
    cfg_dict = cfg.to_dict()
    assert cfg.n_splits == 5, "Default n_splits should be 5"
    assert "data_path" in cfg_dict, "TrainingConfig.to_dict() failed"

    metrics = EvaluationMetrics(auroc=0.93, auprc=0.82, sensitivity=0.76, specificity=0.96)
    metrics_dict = metrics.to_dict()
    assert metrics_dict["auroc"] == 0.93, "EvaluationMetrics.to_dict() failed"
    print("  [PASS] TrainingConfig and EvaluationMetrics schemas verified")

    # 2. Patient-Level GroupKFold Splitter
    print("\n[*] 2. Testing Patient-Level GroupKFold Splitter (Leakage Guard)...")
    df_mock = pd.DataFrame({
        "subject_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        "stay_id": [10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60],
        "feature1": np.random.randn(12),
        "deterioration_next_4h": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    })
    splitter = PatientGroupKFoldSplitter(n_splits=3)
    splits = splitter.split(df_mock, group_col="subject_id")
    assert len(splits) == 3, "Expected 3 folds"

    for fold, (trn_idx, val_idx) in enumerate(splits, 1):
        trn_subs = set(df_mock.iloc[trn_idx]["subject_id"].unique())
        val_subs = set(df_mock.iloc[val_idx]["subject_id"].unique())
        leakage = trn_subs.intersection(val_subs)
        assert len(leakage) == 0, f"Patient leakage detected: {leakage}"
        print(f"  [PASS] Fold {fold}: Train subjects {sorted(list(trn_subs))} | Val subjects {sorted(list(val_subs))} (Overlap: 0)")

    # 3. Clinical Metrics Evaluator
    print("\n[*] 3. Testing Clinical Metrics Evaluator...")
    evaluator = ClinicalMetricsEvaluator()
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.92, 0.12, 0.88, 0.08, 0.85, 0.15])
    m = evaluator.evaluate(y_true, y_prob, threshold=0.5)

    assert m.auroc == 1.0, f"Expected AUROC 1.0, got {m.auroc}"
    assert m.sensitivity == 1.0, f"Expected Sensitivity 1.0, got {m.sensitivity}"
    assert m.specificity == 1.0, f"Expected Specificity 1.0, got {m.specificity}"
    print(f"  [PASS] AUROC: {m.auroc:.4f} | AUPRC: {m.auprc:.4f} | Sens: {m.sensitivity:.4f} | Spec: {m.specificity:.4f} | Brier: {m.brier:.4f}")

    # 4. Model Estimator Wrappers (sklearn, XGBoost, LightGBM)
    print("\n[*] 4. Testing Framework Estimator Wrappers...")
    X_sample = np.random.randn(40, 4)
    y_sample = (X_sample[:, 0] + X_sample[:, 1] > 0).astype(int)

    wrappers = {
        "Logistic Regression": SklearnEstimatorWrapper(LogisticRegression()),
        "Random Forest": SklearnEstimatorWrapper(RandomForestClassifier(n_estimators=10, random_state=42)),
        "XGBoost": XGBoostEstimatorWrapper(n_estimators=10, max_depth=3, random_state=42),
        "LightGBM": LightGBMEstimatorWrapper(n_estimators=10, max_depth=3, random_state=42),
    }

    for name, wrapper in wrappers.items():
        wrapper.fit(X_sample, y_sample)
        preds = wrapper.predict(X_sample)
        probs = wrapper.predict_proba(X_sample)
        assert len(preds) == len(y_sample), f"{name}: predictions length mismatch"
        assert probs.shape == (len(y_sample), 2), f"{name}: predict_proba shape mismatch"
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(y_sample)), atol=1e-5)
        print(f"  [PASS] {name:<20}: fit, predict, and predict_proba contracts validated")

    # 5. RespiratoryPredictorSuite Orchestration
    print("\n[*] 5. Testing High-Level Orchestrator (RespiratoryPredictorSuite)...")
    suite = RespiratoryPredictorSuite()
    suite.build_default_estimators()
    registered = suite.list_estimators()
    expected = ["Logistic Regression", "Random Forest", "LightGBM", "XGBoost"]
    for exp in expected:
        assert exp in registered, f"Missing registered estimator: {exp}"
    print(f"  [PASS] Estimator registry contains all 4 standard models: {registered}")

    print("\n" + "=" * 65)
    print("[SUCCESS] All model training abstractions verified successfully!")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(verify_abstractions())
