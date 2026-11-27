"""
Unit tests for exploratory comorbidity analysis (scripts/eda_comorbidities.py).
Verifies:
- ICD condition mapping logic (text regex and code prefixes)
- Clinical risk flag prevalence calculations
- Total diagnoses per stay distribution statistics
- Co-occurrence matrix calculations
- Edge case handling (empty dataframes, missing columns, duplicate diagnoses within stay)
- Markdown report generation
- Integration pipeline execution on MIMIC-IV demo dataset
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
import pandas as pd

# Add project root and scripts directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eda_comorbidities import (
    CONDITION_DEFINITIONS,
    compute_comorbidity_metrics,
    format_terminal_output,
    generate_markdown_summary,
    load_diagnosis_data,
    map_icd_conditions,
    resolve_table_path,
    run_comorbidity_eda,
)


class TestEDAComorbidities(unittest.TestCase):
    """Test suite for comorbidity mapping, statistics, co-occurrence, and pipeline."""

    def setUp(self):
        # Synthetic diagnosis table
        self.sample_diag = pd.DataFrame({
            "subject_id": [101, 101, 101, 102, 102, 103],
            "hadm_id": [2001, 2001, 2001, 2002, 2002, 2003],
            "seq_num": [1, 2, 3, 1, 2, 1],
            "icd_code": ["51881", "486", "496", "J80", "A419", "V707"],
            "icd_version": [9, 9, 9, 10, 10, 9],
        })

        # Synthetic dictionary table
        self.sample_d_diag = pd.DataFrame({
            "icd_code": ["51881", "486", "496", "J80", "A419", "V707"],
            "icd_version": [9, 9, 9, 10, 10, 9],
            "long_title": [
                "Acute respiratory failure",
                "Pneumonia, organism unspecified",
                "Chronic airway obstruction, not elsewhere classified",
                "Acute respiratory distress syndrome",
                "Sepsis, unspecified organism",
                "Examination of normal infant or child",
            ],
        })

    def test_map_icd_conditions_text_and_codes(self):
        """Verifies that all 5 clinical risk flags are correctly mapped."""
        mapped = map_icd_conditions(self.sample_diag, self.sample_d_diag)

        self.assertEqual(len(mapped), 6)
        # Row 0: Acute respiratory failure
        self.assertEqual(mapped.loc[0, "acute_respiratory_failure"], 1)
        self.assertEqual(mapped.loc[0, "pneumonia"], 0)

        # Row 1: Pneumonia
        self.assertEqual(mapped.loc[1, "pneumonia"], 1)

        # Row 2: COPD
        self.assertEqual(mapped.loc[2, "copd"], 1)

        # Row 3: ARDS
        self.assertEqual(mapped.loc[3, "ards"], 1)

        # Row 4: Sepsis
        self.assertEqual(mapped.loc[4, "sepsis"], 1)

        # Row 5: Normal exam (none of the 5 conditions)
        for cfg in CONDITION_DEFINITIONS:
            self.assertEqual(mapped.loc[5, cfg["key"]], 0)

    def test_map_icd_conditions_code_prefix_fallback_when_title_missing(self):
        """Verifies mapping succeeds even when long_title is missing/empty using code prefix."""
        diag_no_title = pd.DataFrame({
            "subject_id": [201, 202, 203],
            "hadm_id": [3001, 3002, 3003],
            "icd_code": ["J96.01", "482.1", "038.9"],
            "icd_version": [10, 9, 9],
        })
        mapped = map_icd_conditions(diag_no_title, pd.DataFrame())

        # J9601 -> acute_respiratory_failure
        self.assertEqual(mapped.loc[0, "acute_respiratory_failure"], 1)
        # 4821 -> pneumonia
        self.assertEqual(mapped.loc[1, "pneumonia"], 1)
        # 0389 -> sepsis
        self.assertEqual(mapped.loc[2, "sepsis"], 1)

    def test_prevalence_calculations(self):
        """Verifies stay and patient prevalence counts and percentages."""
        mapped = map_icd_conditions(self.sample_diag, self.sample_d_diag)
        metrics = compute_comorbidity_metrics(mapped)

        self.assertEqual(metrics["total_records"], 6)
        self.assertEqual(metrics["total_stays"], 3)
        self.assertEqual(metrics["total_patients"], 3)

        prev_map = {p["key"]: p for p in metrics["prevalence"]}

        # Hadm 2001 (Stay 1) has Resp Fail, Pneumonia, COPD
        # Hadm 2002 (Stay 2) has ARDS, Sepsis
        # Hadm 2003 (Stay 3) has None
        self.assertEqual(prev_map["acute_respiratory_failure"]["stay_count"], 1)
        self.assertAlmostEqual(prev_map["acute_respiratory_failure"]["stay_pct"], 33.33, places=1)
        self.assertEqual(prev_map["ards"]["stay_count"], 1)
        self.assertEqual(prev_map["pneumonia"]["stay_count"], 1)
        self.assertEqual(prev_map["copd"]["stay_count"], 1)
        self.assertEqual(prev_map["sepsis"]["stay_count"], 1)

    def test_cooccurrence_matrix(self):
        """Verifies pairwise co-occurrence counts between condition pairs."""
        mapped = map_icd_conditions(self.sample_diag, self.sample_d_diag)
        metrics = compute_comorbidity_metrics(mapped)

        cooc = metrics["cooccurrence_counts"]
        # In Hadm 2001, Resp Fail co-occurs with Pneumonia and COPD
        self.assertEqual(cooc.loc["acute_respiratory_failure", "pneumonia"], 1)
        self.assertEqual(cooc.loc["acute_respiratory_failure", "copd"], 1)
        self.assertEqual(cooc.loc["acute_respiratory_failure", "ards"], 0)
        self.assertEqual(cooc.loc["acute_respiratory_failure", "sepsis"], 0)

        # In Hadm 2002, ARDS co-occurs with Sepsis
        self.assertEqual(cooc.loc["ards", "sepsis"], 1)
        self.assertEqual(cooc.loc["ards", "pneumonia"], 0)

    def test_stay_diagnosis_count_distribution(self):
        """Verifies distribution statistics of total diagnoses per stay."""
        mapped = map_icd_conditions(self.sample_diag, self.sample_d_diag)
        metrics = compute_comorbidity_metrics(mapped)

        stats = metrics["stay_diagnosis_stats"]
        # Stays have 3, 2, 1 diagnoses -> count=3, mean=2.0, median=2.0, min=1.0, max=3.0
        self.assertEqual(stats["count"], 3)
        self.assertAlmostEqual(stats["mean"], 2.0)
        self.assertAlmostEqual(stats["median"], 2.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 3.0)

    def test_duplicate_diagnoses_within_stay(self):
        """Verifies duplicate condition occurrences within a single stay are deduplicated."""
        dup_diag = pd.DataFrame({
            "subject_id": [101, 101, 101],
            "hadm_id": [2001, 2001, 2001],
            "seq_num": [1, 2, 3],
            "icd_code": ["486", "4821", "480"],  # 3 distinct pneumonia codes in same stay
            "icd_version": [9, 9, 9],
        })
        mapped = map_icd_conditions(dup_diag, pd.DataFrame())
        metrics = compute_comorbidity_metrics(mapped)

        prev_map = {p["key"]: p for p in metrics["prevalence"]}
        # Should count as exactly 1 stay for pneumonia
        self.assertEqual(prev_map["pneumonia"]["stay_count"], 1)
        self.assertEqual(metrics["total_stays"], 1)

    def test_empty_dataframe_handling(self):
        """Verifies empty dataframes do not raise exceptions and return valid structures."""
        empty_diag = pd.DataFrame(columns=["subject_id", "hadm_id", "icd_code", "icd_version"])
        empty_d = pd.DataFrame(columns=["icd_code", "icd_version", "long_title"])

        mapped = map_icd_conditions(empty_diag, empty_d)
        self.assertTrue(mapped.empty)

        metrics = compute_comorbidity_metrics(mapped)
        self.assertEqual(metrics["total_records"], 0)
        self.assertEqual(metrics["total_stays"], 0)
        self.assertEqual(metrics["stay_diagnosis_stats"]["count"], 0)

        # Terminal output formatting handles empty metrics gracefully
        out_str = format_terminal_output(metrics)
        self.assertIn("MIMIC-IV EXPLORATORY COMORBIDITY ANALYSIS REPORT", out_str)

    def test_markdown_generation_and_export(self):
        """Verifies markdown report string generation and file persistence."""
        mapped = map_icd_conditions(self.sample_diag, self.sample_d_diag)
        metrics = compute_comorbidity_metrics(mapped)

        temp_out = PROJECT_ROOT / "outputs" / "test_comorbidity_summary.md"
        try:
            content = generate_markdown_summary(metrics, output_path=temp_out)
            self.assertIn("# 📊 MIMIC-IV Clinical Database - Comorbidity Analysis Summary", content)
            self.assertIn("## 2. Clinical Risk Flag Prevalence", content)
            self.assertIn("## 4. Pairwise Comorbidity Co-Occurrence Matrix", content)
            self.assertTrue(temp_out.is_file())
        finally:
            if temp_out.is_file():
                temp_out.unlink()

    def test_pipeline_on_demo_dataset(self):
        """Integration test on MIMIC-IV demo dataset files."""
        from eda_comorbidities import REL_DATASET_DIR

        base_dir = PROJECT_ROOT / REL_DATASET_DIR
        if not base_dir.is_dir():
            self.skipTest("Demo dataset directory not present")

        df_diag, df_d_diag = load_diagnosis_data(base_dir)
        self.assertGreater(len(df_diag), 0)
        self.assertGreater(len(df_d_diag), 0)

        success = run_comorbidity_eda(
            dataset_dir=base_dir,
            save_markdown=True,
            quiet=True,
        )
        self.assertTrue(success)

        # Check outputs/comorbidity_summary.md was generated
        out_file = PROJECT_ROOT / "outputs" / "comorbidity_summary.md"
        self.assertTrue(out_file.is_file())
        text = out_file.read_text(encoding="utf-8")
        self.assertIn("Hospital Patient Stays", text)
        self.assertIn("Acute Respiratory Failure", text)
        self.assertIn("Pairwise Comorbidity Co-Occurrence Matrix", text)


if __name__ == "__main__":
    unittest.main()
