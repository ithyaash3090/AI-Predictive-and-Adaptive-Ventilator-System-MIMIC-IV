"""
Unit tests for model training abstractions, interfaces, and cross-validation contracts.
Verifies BaseModelEstimator wrappers, metrics evaluators, patient-level GroupKFold splitter,
and dataclass schemas in src/models/trainer.py.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.trainer import (
    BaseModelEstimator,
    BaseFeatureExtractor,
    BaseMetricsEvaluator,
    BaseCrossValidationSplitter,
    TrainingConfig,
    EvaluationMetrics,
    FoldResult,
    BenchmarkSummary,
    SklearnEstimatorWrapper,
    XGBoostEstimatorWrapper,
    LightGBMEstimatorWrapper,
    PatientGroupKFoldSplitter,
    ClinicalMetricsEvaluator,
    RespiratoryPredictorSuite,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class TestTrainerSchemas(unittest.TestCase):
    """Test schema dataclasses and default configurations."""

    def test_training_config_defaults(self):
        cfg = TrainingConfig()
        self.assertEqual(cfg.n_splits, 5)
        self.assertEqual(cfg.classification_threshold, 0.5)
        self.assertEqual(cfg.target_column, "deterioration_next_4h")
        self.assertEqual(cfg.group_column, "subject_id")
        d = cfg.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("data_path", d)

    def test_evaluation_metrics_to_dict(self):
        metrics = EvaluationMetrics(
            auroc=0.92,
            auprc=0.81,
            sensitivity=0.85,
            specificity=0.95,
            precision=0.75,
            f1=0.80,
            brier=0.05,
            true_positives=17,
            false_positives=3,
            true_negatives=76,
            false_negatives=4,
        )
        d = metrics.to_dict()
        self.assertEqual(d["auroc"], 0.92)
        self.assertEqual(d["true_positives"], 17)
        self.assertEqual(d["brier"], 0.05)


class TestPatientGroupKFoldSplitter(unittest.TestCase):
    """Test strict patient-level grouping and leakage prevention."""

    def setUp(self):
        # Create synthetic cohort: 10 subjects, 10 records each (100 total rows)
        subjects = np.repeat(np.arange(1, 11), 10)
        self.df = pd.DataFrame({
            "subject_id": subjects,
            "stay_id": subjects * 100,
            "hour_from_intime": np.tile(np.arange(10), 10),
            "feature1": np.random.randn(100),
            "deterioration_next_4h": np.random.randint(0, 2, size=100),
        })

    def test_strict_patient_isolation(self):
        splitter = PatientGroupKFoldSplitter(n_splits=5)
        splits = splitter.split(self.df, group_col="subject_id")

        self.assertEqual(len(splits), 5)
        all_val_subjects = []

        for train_idx, val_idx in splits:
            train_sub = set(self.df.iloc[train_idx]["subject_id"].unique())
            val_sub = set(self.df.iloc[val_idx]["subject_id"].unique())

            # Verification of zero patient leakage
            intersection = train_sub.intersection(val_sub)
            self.assertEqual(
                len(intersection),
                0,
                f"Data leakage detected with overlapping subjects: {intersection}",
            )
            all_val_subjects.extend(list(val_sub))

        # Each subject should appear in validation exactly once
        self.assertEqual(sorted(all_val_subjects), sorted(list(self.df["subject_id"].unique())))

    def test_leakage_detector_raises_on_overlap(self):
        train_idx = np.array([0, 1, 2, 10])  # Contains subject 1 and subject 2
        val_idx = np.array([3, 4, 11])      # Contains subject 1 and subject 2
        with self.assertRaises(ValueError):
            PatientGroupKFoldSplitter.verify_no_leakage(self.df, "subject_id", train_idx, val_idx)


class TestClinicalMetricsEvaluator(unittest.TestCase):
    """Test clinical performance metrics calculations."""

    def setUp(self):
        self.evaluator = ClinicalMetricsEvaluator()

    def test_perfect_predictions(self):
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        y_prob = np.array([0.95, 0.90, 0.10, 0.05, 0.88, 0.12, 0.92, 0.08])

        metrics = self.evaluator.evaluate(y_true, y_prob, threshold=0.5)

        self.assertEqual(metrics.auroc, 1.0)
        self.assertEqual(metrics.auprc, 1.0)
        self.assertEqual(metrics.sensitivity, 1.0)
        self.assertEqual(metrics.specificity, 1.0)
        self.assertEqual(metrics.f1, 1.0)
        self.assertLess(metrics.brier, 0.05)

    def test_two_dimensional_probabilities(self):
        y_true = np.array([1, 0, 1, 0])
        # [P(0), P(1)]
        y_prob = np.array([
            [0.2, 0.8],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.85, 0.15],
        ])

        metrics = self.evaluator.evaluate(y_true, y_prob, threshold=0.5)
        self.assertEqual(metrics.sensitivity, 1.0)
        self.assertEqual(metrics.specificity, 1.0)


class TestModelEstimatorWrappers(unittest.TestCase):
    """Test BaseModelEstimator contract and framework wrappers."""

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(80, 5)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(int)

    def test_cannot_instantiate_abstract_base_estimator(self):
        with self.assertRaises(TypeError):
            BaseModelEstimator()

    def test_sklearn_wrapper_logistic_regression(self):
        wrapper = SklearnEstimatorWrapper(LogisticRegression())
        wrapper.fit(self.X, self.y)

        preds = wrapper.predict(self.X)
        probs = wrapper.predict_proba(self.X)

        self.assertEqual(len(preds), len(self.y))
        self.assertEqual(probs.shape, (len(self.y), 2))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(self.y)), atol=1e-5)

        importances = wrapper.get_feature_importances()
        self.assertIsNotNone(importances)
        self.assertEqual(len(importances), 5)

    def test_sklearn_wrapper_random_forest(self):
        wrapper = SklearnEstimatorWrapper(RandomForestClassifier(n_estimators=10, random_state=42))
        wrapper.fit(self.X, self.y)

        probs = wrapper.predict_proba(self.X)
        self.assertEqual(probs.shape, (len(self.y), 2))
        importances = wrapper.get_feature_importances()
        self.assertIsNotNone(importances)
        self.assertEqual(len(importances), 5)

    def test_xgboost_wrapper(self):
        wrapper = XGBoostEstimatorWrapper(n_estimators=10, max_depth=2, random_state=42)
        wrapper.fit(self.X, self.y)

        preds = wrapper.predict(self.X)
        probs = wrapper.predict_proba(self.X)

        self.assertEqual(len(preds), len(self.y))
        self.assertEqual(probs.shape, (len(self.y), 2))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(self.y)), atol=1e-5)
        self.assertIsNotNone(wrapper.get_feature_importances())

    def test_lightgbm_wrapper(self):
        wrapper = LightGBMEstimatorWrapper(n_estimators=10, max_depth=2, random_state=42)
        wrapper.fit(self.X, self.y)

        preds = wrapper.predict(self.X)
        probs = wrapper.predict_proba(self.X)

        self.assertEqual(len(preds), len(self.y))
        self.assertEqual(probs.shape, (len(self.y), 2))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(self.y)), atol=1e-5)
        self.assertIsNotNone(wrapper.get_feature_importances())


class TestRespiratoryPredictorSuite(unittest.TestCase):
    """Test master suite orchestration contracts."""

    def test_suite_estimator_registration(self):
        suite = RespiratoryPredictorSuite()
        lr_wrapper = SklearnEstimatorWrapper(LogisticRegression())
        suite.register_estimator("test_lr", lr_wrapper)

        self.assertIn("test_lr", suite.list_estimators())
        retrieved = suite.get_estimator("test_lr")
        self.assertIs(retrieved, lr_wrapper)

    def test_suite_register_invalid_type_raises(self):
        suite = RespiratoryPredictorSuite()
        with self.assertRaises(TypeError):
            suite.register_estimator("invalid", "not_an_estimator")

    def test_suite_build_default_estimators(self):
        suite = RespiratoryPredictorSuite()
        suite.build_default_estimators()
        expected = ["Logistic Regression", "Random Forest", "LightGBM", "XGBoost"]
        for model in expected:
            self.assertIn(model, suite.list_estimators())


if __name__ == "__main__":
    unittest.main()
