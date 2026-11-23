"""
Unit tests for MIMIC-IV demographic exploratory data analysis (scripts/eda_demographics.py).
Verifies patient count, gender distribution, age statistics (mean, median, IQR, min, max),
race/ethnicity and admission type breakdowns, and markdown generation.
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

from eda_demographics import (
    REL_DATASET_DIR,
    compute_admission_demographics,
    compute_patient_demographics,
    format_terminal_output,
    generate_markdown_summary,
    load_demographic_data,
    resolve_table_path,
    run_demographics_eda,
)


class TestEDADemographics(unittest.TestCase):
    """Test suite for demographic calculations, formatting, and pipeline."""

    def setUp(self):
        # Synthetic patients dataframe
        self.sample_patients = pd.DataFrame({
            "subject_id": [1001, 1002, 1003, 1004, 1005],
            "gender": ["M", "F", "M", "M", "F"],
            "anchor_age": [20, 30, 40, 50, 60],
            "anchor_year": [2150, 2151, 2152, 2153, 2154],
        })

        # Synthetic admissions dataframe
        self.sample_admissions = pd.DataFrame({
            "subject_id": [1001, 1001, 1002, 1003, 1004],
            "hadm_id": [20001, 20002, 20003, 20004, 20005],
            "race": ["WHITE", "WHITE", "BLACK/AFRICAN AMERICAN", "ASIAN", "UNKNOWN"],
            "admission_type": [
                "EW EMER.",
                "OBSERVATION ADMIT",
                "EW EMER.",
                "URGENT",
                "ELECTIVE",
            ],
        })

    def test_compute_patient_demographics(self):
        """Verifies unique patient count, gender ratios, and age IQR metrics."""
        demo = compute_patient_demographics(self.sample_patients)

        self.assertEqual(demo["unique_patients"], 5)
        self.assertEqual(demo["total_records"], 5)

        # Check gender
        gender_map = {g["gender"]: g for g in demo["gender_distribution"]}
        self.assertEqual(gender_map["M"]["count"], 3)
        self.assertEqual(gender_map["M"]["percentage"], 60.0)
        self.assertEqual(gender_map["F"]["count"], 2)
        self.assertEqual(gender_map["F"]["percentage"], 40.0)

        # Check age statistics
        # values: [20, 30, 40, 50, 60] -> mean=40, median=40, q25=30, q75=50, iqr=20, min=20, max=60
        age = demo["age_statistics"]
        self.assertEqual(age["count"], 5)
        self.assertAlmostEqual(age["mean"], 40.0)
        self.assertAlmostEqual(age["median"], 40.0)
        self.assertAlmostEqual(age["q25"], 30.0)
        self.assertAlmostEqual(age["q75"], 50.0)
        self.assertAlmostEqual(age["iqr"], 20.0)
        self.assertEqual(age["min"], 20.0)
        self.assertEqual(age["max"], 60.0)

    def test_compute_admission_demographics(self):
        """Verifies race and admission type distributions across merged admissions."""
        admit_demo = compute_admission_demographics(
            self.sample_patients, self.sample_admissions
        )

        self.assertEqual(admit_demo["total_admissions"], 5)
        self.assertEqual(admit_demo["unique_admitted_patients"], 4)

        # Race breakdown
        race_map = {r["race"]: r for r in admit_demo["race_distribution"]}
        self.assertIn("WHITE", race_map)
        self.assertEqual(race_map["WHITE"]["count"], 2)
        self.assertEqual(race_map["WHITE"]["percentage"], 40.0)

        # Admission type breakdown
        admit_map = {
            a["admission_type"]: a for a in admit_demo["admission_type_distribution"]
        }
        self.assertIn("EW EMER.", admit_map)
        self.assertEqual(admit_map["EW EMER."]["count"], 2)
        self.assertEqual(admit_map["EW EMER."]["percentage"], 40.0)

    def test_format_terminal_output(self):
        """Ensures terminal formatter includes required headers and keywords."""
        p_demo = compute_patient_demographics(self.sample_patients)
        a_demo = compute_admission_demographics(
            self.sample_patients, self.sample_admissions
        )

        text = format_terminal_output(p_demo, a_demo)
        self.assertIn("MIMIC-IV DEMOGRAPHIC SUMMARY REPORT", text)
        self.assertIn("GENDER DISTRIBUTION", text)
        self.assertIn("AGE / ANCHOR AGE STATISTICS", text)
        self.assertIn("RACE & ETHNICITY BREAKDOWN", text)
        self.assertIn("ADMISSION TYPE BREAKDOWN", text)

    def test_generate_markdown_summary(self):
        """Ensures markdown summary generates clean tables and writes to file."""
        p_demo = compute_patient_demographics(self.sample_patients)
        a_demo = compute_admission_demographics(
            self.sample_patients, self.sample_admissions
        )

        temp_out = PROJECT_ROOT / "outputs" / "test_demographic_summary.md"
        try:
            md_content = generate_markdown_summary(
                p_demo, a_demo, output_path=temp_out
            )
            self.assertIn("# 📊 MIMIC-IV Clinical Database - Demographic Summary", md_content)
            self.assertIn("## 2. Gender Distribution", md_content)
            self.assertIn("## 3. Patient Age & Anchor Age Statistics", md_content)
            self.assertTrue(temp_out.is_file())
        finally:
            if temp_out.is_file():
                temp_out.unlink()

    def test_pipeline_on_mimic_demo_dataset(self):
        """Integration test on actual demo dataset."""
        base_dir = PROJECT_ROOT / REL_DATASET_DIR
        if not base_dir.is_dir():
            self.skipTest(f"Demo dataset directory not present at {base_dir}")

        p_path = resolve_table_path(base_dir, "hosp/patients.csv.gz")
        self.assertIsNotNone(p_path)

        success = run_demographics_eda(
            dataset_dir=base_dir,
            save_markdown=True,
            quiet=True,
        )
        self.assertTrue(success)

        # Verify default markdown was written
        out_md = PROJECT_ROOT / "outputs" / "demographic_summary.md"
        self.assertTrue(out_md.is_file())
        content = out_md.read_text(encoding="utf-8")
        self.assertIn("Unique Patients", content)
        self.assertIn("Gender Distribution", content)
        self.assertIn("Mean Age", content)


if __name__ == "__main__":
    unittest.main()
