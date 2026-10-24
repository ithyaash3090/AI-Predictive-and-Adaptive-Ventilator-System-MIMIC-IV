"""
Train and benchmark all models using GroupKFold patient-level cross validation.
"""
import os
import sys

sys.path.insert(0, os.path.abspath('.'))

from src.models.trainer import RespiratoryPredictorSuite

def main():
    suite = RespiratoryPredictorSuite()
    results = suite.train_and_benchmark(n_splits=5)
    
    print("\n" + "="*70)
    print("         MIMIC-IV RESPIRATORY DETERIORATION BENCHMARK RESULTS")
    print("="*70)
    for model_name in ["Logistic Regression", "Random Forest", "LightGBM", "XGBoost"]:
        m = results[model_name]
        print(f"\nModel: {model_name}")
        print(f"  AUROC       : {m['auroc']:.4f} (+/- {m['auroc_std']:.4f})")
        print(f"  AUPRC       : {m['auprc']:.4f} (+/- {m['auprc_std']:.4f})")
        print(f"  Sensitivity : {m['sensitivity']:.4f}")
        print(f"  Specificity : {m['specificity']:.4f}")
        print(f"  Precision   : {m['precision']:.4f}")
        print(f"  F1-Score    : {m['f1']:.4f}")
        print(f"  Brier Score : {m['brier']:.4f}")

    print("\n" + "-"*70)
    print(f"Average Early Warning Lead Time: {results.get('average_warning_lead_time_hours', 'N/A')} hours")
    print("\nTop 10 Clinical Features Driving Deterioration Risk:")
    for i, f in enumerate(results.get("top_features", [])[:10], 1):
        print(f"  {i}. {f['feature']} (Relative Importance: {f['importance']:.4f})")
    print("="*70)

if __name__ == "__main__":
    main()
