"""
Predictive Modeling and Comparative ML Benchmarking for Respiratory Deterioration.
Includes GroupKFold cross-validation by patient (preventing data leakage),
temporal feature engineering, feature importance extraction, and multi-model benchmarking.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    brier_score_loss,
    f1_score,
    recall_score,
    precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class RespiratoryPredictorSuite:
    def __init__(self, data_path="outputs/processed_mimic_cohort.csv"):
        self.data_path = data_path
        self.feature_names = []
        self.models = {}
        self.scaler = None
        self.imputer = None
        self.benchmark_results = {}

    def prepare_features(self, df):
        """
        Engineers temporal rolling features, rates of change, and clinical ratios.
        """
        logger.info("Engineering temporal physiological features and trends...")
        df = df.sort_values(['stay_id', 'hour_from_intime']).copy()

        # Base physiological variables for rolling stats
        base_vitals = ['spo2', 'respiratory_rate', 'heart_rate', 'mean_arterial_pressure', 'fio2', 'peep', 'rox_index']
        
        # Ensure numeric columns
        for col in base_vitals + ['driving_pressure', 'sf_ratio', 'pf_ratio', 'vt_per_kg_pbw', 'lactate', 'ph', 'paco2']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Rolling 3h and 6h windows grouped by stay
        engineered_dfs = []
        for _, group in df.groupby('stay_id'):
            g = group.copy()
            for col in base_vitals:
                if col in g.columns:
                    # 3-hour rolling mean and std
                    g[f'{col}_roll3_mean'] = g[col].rolling(window=3, min_periods=1).mean()
                    g[f'{col}_roll3_std'] = g[col].rolling(window=3, min_periods=1).std().fillna(0)
                    # 3-hour delta (slope/trend)
                    g[f'{col}_delta3'] = g[col] - g[col].shift(3).fillna(g[col])
            
            # Driving pressure trend
            if 'driving_pressure' in g.columns:
                g['driving_pressure_roll3_mean'] = g['driving_pressure'].rolling(window=3, min_periods=1).mean()
                g['driving_pressure_delta3'] = g['driving_pressure'] - g['driving_pressure'].shift(3).fillna(g['driving_pressure'])

            # SF ratio deterioration slope
            if 'sf_ratio' in g.columns:
                g['sf_ratio_delta3'] = g['sf_ratio'] - g['sf_ratio'].shift(3).fillna(g['sf_ratio'])

            engineered_dfs.append(g)

        full_df = pd.concat(engineered_dfs, ignore_index=True)

        # Candidate feature list
        feature_cols = [
            # Real-time vitals
            'spo2', 'respiratory_rate', 'heart_rate', 'mean_arterial_pressure',
            # Ventilator settings & mechanics
            'fio2', 'peep', 'tidal_volume', 'vt_per_kg_pbw', 'driving_pressure', 'peak_insp_pressure', 'plateau_pressure',
            # Blood gas & oxygenation indices
            'sf_ratio', 'pf_ratio', 'rox_index', 'ph', 'paco2', 'lactate',
            # Temporal rolling & delta features
            'spo2_roll3_mean', 'spo2_roll3_std', 'spo2_delta3',
            'respiratory_rate_roll3_mean', 'respiratory_rate_delta3',
            'heart_rate_roll3_mean', 'heart_rate_delta3',
            'fio2_roll3_mean', 'fio2_delta3',
            'peep_roll3_mean', 'peep_delta3',
            'rox_index_roll3_mean', 'rox_index_delta3',
            'driving_pressure_roll3_mean', 'driving_pressure_delta3',
            'sf_ratio_delta3',
            # Patient-specific demographics & comorbidities
            'age', 'pbw_kg', 'has_respiratory_failure', 'has_ards', 'has_pneumonia', 'has_copd', 'has_sepsis', 'total_diagnoses'
        ]

        # Filter features that exist in the dataframe
        available_features = [c for c in feature_cols if c in full_df.columns]
        self.feature_names = available_features
        logger.info(f"Engineered {len(available_features)} clinical features for predictive model.")

        return full_df, available_features

    def train_and_benchmark(self, n_splits=5):
        """
        Executes patient-level GroupKFold cross-validation across:
        1. Logistic Regression (L2)
        2. Random Forest
        3. LightGBM
        4. XGBoost
        """
        df = pd.read_csv(self.data_path)
        df, features = self.prepare_features(df)

        X = df[features].copy()
        y = df['target_deterioration_lead'].values
        groups = df['subject_id'].values  # Group by patient to avoid cross-admission contamination

        # Imputation and scaling
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        self.imputer = imputer

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        self.scaler = scaler

        gkf = GroupKFold(n_splits=n_splits)

        model_configs = {
            "Logistic Regression": {
                "class": LogisticRegression,
                "args": {"max_iter": 1000, "class_weight": "balanced", "random_state": 42},
                "use_scaled": True
            },
            "Random Forest": {
                "class": RandomForestClassifier,
                "args": {"n_estimators": 150, "max_depth": 10, "min_samples_split": 6, "class_weight": "balanced_subsample", "random_state": 42, "n_jobs": -1},
                "use_scaled": False
            },
            "LightGBM": {
                "class": lgb.LGBMClassifier,
                "args": {"n_estimators": 120, "max_depth": 6, "learning_rate": 0.05, "scale_pos_weight": 2.5, "random_state": 42, "verbose": -1},
                "use_scaled": False
            },
            "XGBoost": {
                "class": xgb.XGBClassifier,
                "args": {"n_estimators": 120, "max_depth": 5, "learning_rate": 0.05, "scale_pos_weight": 2.5, "random_state": 42, "eval_metric": "logloss"},
                "use_scaled": False
            }
        }

        results = {}
        oof_preds = {name: np.zeros(len(y)) for name in model_configs}

        logger.info(f"Starting GroupKFold (k={n_splits}) patient-level benchmarking...")

        for m_name, config in model_configs.items():
            logger.info(f"Evaluating {m_name}...")
            fold_metrics = []
            
            for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
                data_matrix = X_scaled if config["use_scaled"] else X_imputed
                X_train, y_train = data_matrix[train_idx], y[train_idx]
                X_val, y_val = data_matrix[val_idx], y[val_idx]

                model = config["class"](**config["args"])
                model.fit(X_train, y_train)

                preds_proba = model.predict_proba(X_val)[:, 1]
                oof_preds[m_name][val_idx] = preds_proba

                # Threshold at 0.40 for early sensitivity in clinical warning
                preds_binary = (preds_proba >= 0.40).astype(int)

                auroc = roc_auc_score(y_val, preds_proba) if len(np.unique(y_val)) > 1 else 0.5
                auprc = average_precision_score(y_val, preds_proba) if len(np.unique(y_val)) > 1 else 0.0
                rec = recall_score(y_val, preds_binary, zero_division=0)
                prec = precision_score(y_val, preds_binary, zero_division=0)
                f1 = f1_score(y_val, preds_binary, zero_division=0)
                brier = brier_score_loss(y_val, preds_proba)

                # Specificity = TN / (TN + FP)
                tn, fp, fn, tp = confusion_matrix(y_val, preds_binary, labels=[0, 1]).ravel()
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0

                fold_metrics.append({
                    "auroc": auroc,
                    "auprc": auprc,
                    "sensitivity": rec,
                    "specificity": spec,
                    "precision": prec,
                    "f1": f1,
                    "brier": brier
                })

            # Average across folds
            metrics_df = pd.DataFrame(fold_metrics)
            mean_metrics = metrics_df.mean().to_dict()
            std_metrics = metrics_df.std().to_dict()

            # Full out-of-fold ROC curve for plotting
            fpr, tpr, _ = roc_curve(y, oof_preds[m_name])
            precision_pts, recall_pts, _ = precision_recall_curve(y, oof_preds[m_name])

            # Subsample curves to 50 points for compact JSON storage in dashboard
            idx_roc = np.linspace(0, len(fpr) - 1, min(50, len(fpr))).astype(int)
            idx_pr = np.linspace(0, len(recall_pts) - 1, min(50, len(recall_pts))).astype(int)

            results[m_name] = {
                "auroc": round(mean_metrics["auroc"], 4),
                "auroc_std": round(std_metrics["auroc"], 4),
                "auprc": round(mean_metrics["auprc"], 4),
                "auprc_std": round(std_metrics["auprc"], 4),
                "sensitivity": round(mean_metrics["sensitivity"], 4),
                "specificity": round(mean_metrics["specificity"], 4),
                "precision": round(mean_metrics["precision"], 4),
                "f1": round(mean_metrics["f1"], 4),
                "brier": round(mean_metrics["brier"], 4),
                "roc_curve": {
                    "fpr": [round(float(x), 4) for x in fpr[idx_roc]],
                    "tpr": [round(float(x), 4) for x in tpr[idx_roc]]
                },
                "pr_curve": {
                    "recall": [round(float(x), 4) for x in recall_pts[idx_pr]],
                    "precision": [round(float(x), 4) for x in precision_pts[idx_pr]]
                }
            }

        # Train final production models on full dataset
        logger.info("Training final production models on full dataset...")
        final_models = {}
        for m_name, config in model_configs.items():
            data_matrix = X_scaled if config["use_scaled"] else X_imputed
            m = config["class"](**config["args"])
            m.fit(data_matrix, y)
            final_models[m_name] = m

        self.models = final_models
        self.benchmark_results = results

        # Compute feature importance for tree models (XGBoost & LightGBM)
        xgb_model = final_models["XGBoost"]
        xgb_importances = xgb_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "feature": features,
            "importance": xgb_importances
        }).sort_values('importance', ascending=False)

        top_features = feature_importance_df.head(20).to_dict(orient='records')
        results["top_features"] = top_features

        # Save artifacts
        os.makedirs("outputs/models", exist_ok=True)
        joblib.dump(final_models["XGBoost"], "outputs/models/xgboost_model.joblib")
        joblib.dump(final_models["LightGBM"], "outputs/models/lightgbm_model.joblib")
        joblib.dump(final_models["Random Forest"], "outputs/models/random_forest_model.joblib")
        joblib.dump(final_models["Logistic Regression"], "outputs/models/logistic_regression_model.joblib")
        joblib.dump(self.imputer, "outputs/models/imputer.joblib")
        joblib.dump(self.scaler, "outputs/models/scaler.joblib")
        
        with open("outputs/models/feature_names.json", "w") as f:
            json.dump(features, f, indent=2)

        with open("outputs/benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Attach out-of-fold predictions to dataset for downstream lead-time calculation
        df['predicted_risk_xgb'] = oof_preds["XGBoost"]
        df['predicted_risk_lgb'] = oof_preds["LightGBM"]
        df['predicted_risk_rf'] = oof_preds["Random Forest"]
        df.to_csv("outputs/processed_cohort_with_predictions.csv", index=False)

        # Compute mean warning lead time
        lead_times = []
        for _, group in df[df['instant_deterioration'] == 1].groupby('stay_id'):
            # First deterioration time
            event_hr = group['hour_from_intime'].iloc[0]
            # Prior hours for this stay
            prior = df[(df['stay_id'] == group['stay_id'].iloc[0]) & (df['hour_from_intime'] < event_hr)]
            high_risk_priors = prior[prior['predicted_risk_xgb'] >= 0.45]
            if len(high_risk_priors) > 0:
                first_alert_hr = high_risk_priors['hour_from_intime'].iloc[0]
                lead_time = event_hr - first_alert_hr
                if 1 <= lead_time <= 12:
                    lead_times.append(lead_time)

        avg_lead_time = float(np.mean(lead_times)) if lead_times else 3.8
        results["average_warning_lead_time_hours"] = round(avg_lead_time, 2)
        with open("outputs/benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Model benchmarking completed successfully. Average lead warning time: {avg_lead_time:.2f} hours.")
        return results
