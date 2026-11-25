"""
Unit tests for the patient-level cross-validation splitter CLI blueprint.
Validates CLI argument parsing, leakage detection, fold partition integrity,
and markdown summary export in scripts/test_splitter.py.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from test_splitter import (
    parse_args,
    load_cohort,
    evaluate_splits,
    format_terminal_report,
    export_markdown_report,
    main,
)


class TestSplitterCLI(unittest.TestCase):
    """Test suite for patient-level splitter blueprint utility."""

    def setUp(self):
        # Create synthetic cohort: 12 patients, 24 stays, 120 hourly rows
        np.random.seed(42)
        subjects = np.repeat(np.arange(1, 13), 10)
        stays = np.repeat(np.arange(101, 125), 5)
        self.mock_df = pd.DataFrame({
            "subject_id": subjects,
            "stay_id": stays,
            "hour_from_intime": np.tile(np.arange(5), 24),
            "feature_val": np.random.randn(120),
            "target_deterioration_lead": np.random.choice([0, 1], size=120, p=[0.8, 0.2]),
        })

    def test_parse_args_defaults(self):
        """Verifies default argument values."""
        args = parse_args([])
        self.assertEqual(args.n_splits, 5)
        self.assertEqual(args.group_col, "subject_id")
        self.assertEqual(args.stay_col, "stay_id")
        self.assertEqual(args.target_col, "target_deterioration_lead")

    def test_parse_args_custom(self):
        """Verifies custom argument parsing."""
        args = parse_args([
            "--n-splits", "3",
            "--group-col", "patient_id",
            "--cohort-path", "custom_path.csv",
        ])
        self.assertEqual(args.n_splits, 3)
        self.assertEqual(args.group_col, "patient_id")
        self.assertEqual(args.cohort_path, "custom_path.csv")

    def test_evaluate_splits_zero_leakage(self):
        """Verifies that evaluate_splits enforces zero patient leakage across all folds."""
        results = evaluate_splits(
            df=self.mock_df,
            n_splits=4,
            group_col="subject_id",
            stay_col="stay_id",
            target_col="target_deterioration_lead",
        )
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertEqual(r["overlap_patients"], 0)
            self.assertEqual(r["leakage_status"], "PASS")
            self.assertGreater(r["val_rows"], 0)
            self.assertGreater(r["train_rows"], 0)

    def test_evaluate_splits_detects_leakage(self):
        """Verifies that forced patient overlap triggers ValueError."""
        # Corrupt data by creating a copy of subject in multiple distinct synthetic partitions
        # Test directly via splitter verify_no_leakage
        from src.models.trainer import PatientGroupKFoldSplitter
        train_idx = np.array([0, 1, 2])    # subject 1
        val_idx = np.array([3, 4, 15])     # subject 1 and subject 2
        with self.assertRaises(ValueError):
            PatientGroupKFoldSplitter.verify_no_leakage(self.mock_df, "subject_id", train_idx, val_idx)

    def test_load_cohort_handles_fallback_target_alias(self):
        """Verifies that load_cohort falls back to deterioration_next_4h if canonical column is missing."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df_alias = self.mock_df.rename(columns={"target_deterioration_lead": "deterioration_next_4h"})
            df_alias.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            loaded = load_cohort(temp_path, "target_deterioration_lead")
            self.assertIn("deterioration_next_4h", loaded.columns)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_format_terminal_report_output(self):
        """Verifies terminal report contains headers and pass statuses."""
        results = evaluate_splits(self.mock_df, n_splits=3)
        text = format_terminal_report(
            results=results,
            n_splits=3,
            total_patients=12,
            total_stays=24,
            total_rows=120,
        )
        self.assertIn("MIMIC-IV PATIENT-LEVEL CROSS-VALIDATION SPLITTER BLUEPRINT", text)
        self.assertIn("[SUCCESS]", text)
        self.assertIn("[PASS]", text)

    def test_export_markdown_report_creates_file(self):
        """Verifies export_markdown_report writes valid markdown file."""
        results = evaluate_splits(self.mock_df, n_splits=3)
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            temp_path = f.name

        try:
            export_markdown_report(
                results=results,
                output_path=temp_path,
                total_patients=12,
                total_stays=24,
                total_rows=120,
                n_splits=3,
            )
            self.assertTrue(os.path.exists(temp_path))
            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("# Patient-Level Cross-Validation Splitter Audit Report", content)
            self.assertIn("Fold Partitioning Table", content)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_end_to_end_cli_execution_on_cohort(self):
        """Tests full execution of scripts/test_splitter.py on cohort dataset."""
        cohort_csv = PROJECT_ROOT / "outputs" / "processed_mimic_cohort.csv"
        if not cohort_csv.exists():
            self.skipTest("outputs/processed_mimic_cohort.csv not found")

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            temp_report = f.name

        try:
            ret = main([
                "--cohort-path", str(cohort_csv),
                "--n-splits", "5",
                "--output-report", temp_report,
            ])
            self.assertEqual(ret, 0)
            self.assertTrue(os.path.exists(temp_report))
        finally:
            if os.path.exists(temp_report):
                os.remove(temp_report)


if __name__ == "__main__":
    unittest.main()
