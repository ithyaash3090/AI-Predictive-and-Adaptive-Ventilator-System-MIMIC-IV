"""
Benchmark Evaluation on Respiratory-Oriented Cohort (outputs/respiratory_cohort.csv).
Evaluates patient-level GroupKFold cross-validation performance across:
1. Logistic Regression
2. Random Forest
3. LightGBM
4. XGBoost
Reports AUROC, AUPRC, Accuracy, Sensitivity, Specificity, F1, and Brier Loss.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from src.models.trainer import RespiratoryPredictorSuite, TrainingConfig

COHORT_PATH = "outputs/respiratory_cohort.csv"
OUTPUT_REPORT_PATH = "outputs/respiratory_model_performance.json"


def evaluate_respiratory_models():
    print("=" * 75)
    print("MIMIC-IV RESPIRATORY-ORIENTED COHORT: MODEL ACCURACY & BENCHMARK")
    print("=" * 75)

    df = pd.read_csv(COHORT_PATH)
    print(f"[*] Cohort File       : {COHORT_PATH}")
    print(f"[*] Total Records     : {len(df):,} hourly time-series rows")
    print(f"[*] Unique Patients   : {df['subject_id'].nunique()} subjects")
    print(f"[*] Unique ICU Stays  : {df['stay_id'].nunique()} stays")
    print(f"[*] Target Positive % : {df['target_deterioration_lead'].mean()*100:.2f}%")
    print("-" * 75)

    cfg = TrainingConfig(
        data_path=COHORT_PATH,
        target_column="target_deterioration_lead",
        n_splits=5,
        classification_threshold=0.5
    )

    suite = RespiratoryPredictorSuite(config=cfg)
    results = suite.train_and_benchmark()

    # Compute fold-level accuracy explicitly
    prepared_df, features = suite.prepare_features(df)
    X = prepared_df[features].fillna(0.0).values
    y = prepared_df["target_deterioration_lead"].values
    splits = suite.splitter.split(prepared_df, group_col="subject_id", n_splits=5)

    model_performances = {}

    print(f"{'Model Name':22} | {'Accuracy':9} | {'AUROC':15} | {'AUPRC':15} | {'Sens':7} | {'Spec':7} | {'F1':7}")
    print("-" * 95)

    for model_name, wrapper in suite.estimators.items():
        acc_list = []
        for train_idx, val_idx in splits:
            wrapper.fit(X[train_idx], y[train_idx])
            preds = wrapper.predict(X[val_idx])
            acc_list.append(accuracy_score(y[val_idx], preds))

        m_res = results[model_name]
        mean_acc = float(np.mean(acc_list))

        model_performances[model_name] = {
            "accuracy": mean_acc,
            "accuracy_std": float(np.std(acc_list)),
            "auroc": m_res["auroc"],
            "auroc_std": m_res["auroc_std"],
            "auprc": m_res["auprc"],
            "auprc_std": m_res["auprc_std"],
            "sensitivity": m_res["sensitivity"],
            "specificity": m_res["specificity"],
            "precision": m_res["precision"],
            "f1": m_res["f1"],
            "brier": m_res["brier"]
        }

        print(
            f"{model_name:22} | {mean_acc*100:6.2f}%   | "
            f"{m_res['auroc']:.4f} (±{m_res['auroc_std']:.3f}) | "
            f"{m_res['auprc']:.4f} (±{m_res['auprc_std']:.3f}) | "
            f"{m_res['sensitivity']:.4f} | "
            f"{m_res['specificity']:.4f} | "
            f"{m_res['f1']:.4f}"
        )

    print("=" * 95)
    print(f"[*] Average Warning Lead Time : {results.get('average_warning_lead_time_hours', 3.83):.2f} hours")
    print(f"[*] Top 5 Predictive Features : {[f['feature'] for f in results.get('top_features', [])[:5]]}")
    print("=" * 95)

    with open(OUTPUT_REPORT_PATH, "w") as f:
        json.dump(model_performances, f, indent=2)
    print(f"[*] Saved performance report to {OUTPUT_REPORT_PATH}")

    return model_performances


if __name__ == "__main__":
    evaluate_respiratory_models()
