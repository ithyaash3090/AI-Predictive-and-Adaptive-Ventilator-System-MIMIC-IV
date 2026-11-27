"""
Unit tests for MIMIC-IV laboratory blood gas panel validator (scripts/validate_labs.py).
Tests Item IDs 50820 (pH), 50818 (PaCO2), 50821 (PaO2), and 50813 (Lactate),
validating counts, missingness rates, plausibility envelopes, and markdown output.
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

from validate_labs import (
    ABG_DEFINITIONS,
    REL_DATASET_DIR,
    audit_blood_gas_panel,
    resolve_table_path,
    save_markdown_report,
)


class TestValidateLabs(unittest.TestCase):
    """Test suite for blood gas panel validation logic and metrics."""

    def test_abg_definitions_structure(self):
        """Verifies that all 4 required ABG analytes are configured properly."""
        itemids = [d["itemid"] for d in ABG_DEFINITIONS]
        self.assertIn(50820, itemids, "pH (50820) must be present")
        self.assertIn(50818, itemids, "PaCO2 (50818) must be present")
        self.assertIn(50821, itemids, "PaO2 (50821) must be present")
        self.assertIn(50813, itemids, "Lactate (50813) must be present")
        self.assertEqual(len(itemids), 4, "Must define exactly 4 core ABG items")

    def test_resolve_table_path(self):
        """Tests that resolve_table_path finds labevents in the demo dataset."""
        base_dir = PROJECT_ROOT / REL_DATASET_DIR
        resolved = resolve_table_path(base_dir, "hosp/labevents.csv.gz")
        self.assertIsNotNone(resolved, "hosp/labevents.csv.gz should resolve to an existing file")
        self.assertTrue(resolved.is_file(), f"Resolved path {resolved} must be a file")

    def test_audit_blood_gas_panel_on_demo_dataset(self):
        """Runs the audit on the actual demo dataset and verifies counts and bounds."""
        base_dir = PROJECT_ROOT / REL_DATASET_DIR
        results = audit_blood_gas_panel(base_dir)

        self.assertIn("total_abg_obs", results)
        self.assertGreater(results["total_abg_obs"], 3000, "Should have over 3,000 ABG records")
        self.assertEqual(len(results["item_results"]), 4, "Should have results for all 4 analytes")

        # Check each analyte results
        item_map = {r["itemid"]: r for r in results["item_results"]}
        
        # pH
        ph_res = item_map[50820]
        self.assertEqual(ph_res["null_count"], 0, "pH should have 0 null values")
        self.assertAlmostEqual(ph_res["in_range_pct"], 100.0, places=1, msg="pH should be ~100% in [6.8, 7.8]")

        # PaCO2
        paco2_res = item_map[50818]
        self.assertEqual(paco2_res["null_count"], 0, "PaCO2 should have 0 null values")
        self.assertGreaterEqual(paco2_res["in_range_pct"], 99.0, "PaCO2 in-range should be >= 99%")
        self.assertEqual(paco2_res["high_outlier_count"], 1, "PaCO2 has 1 extreme outlier (123.0 mmHg)")

        # PaO2
        pao2_res = item_map[50821]
        self.assertEqual(pao2_res["null_count"], 0, "PaO2 should have 0 null values")
        self.assertGreaterEqual(pao2_res["in_range_pct"], 97.0, "PaO2 in-range should be >= 97%")
        self.assertGreater(pao2_res["low_outlier_count"], 0, "PaO2 has severe hypoxemia outliers (< 30 mmHg)")

        # Lactate
        lactate_res = item_map[50813]
        self.assertEqual(lactate_res["null_count"], 0, "Lactate should have 0 null values")
        self.assertEqual(lactate_res["in_range_pct"], 100.0, "Lactate should be 100% in [0.3, 25.0]")

    def test_save_markdown_report(self):
        """Verifies that markdown summary report generation works and outputs valid markdown."""
        base_dir = PROJECT_ROOT / REL_DATASET_DIR
        results = audit_blood_gas_panel(base_dir)

        test_out = PROJECT_ROOT / "outputs" / "test_blood_gas_report.md"
        try:
            save_markdown_report(results, test_out)
            self.assertTrue(test_out.exists(), "Report file must exist")
            content = test_out.read_text(encoding="utf-8")
            self.assertIn("# 🩸 Arterial Blood Gas (ABG) Panel Validation Report", content)
            self.assertIn("50820", content)
            self.assertIn("50821", content)
        finally:
            if test_out.exists():
                test_out.unlink()


if __name__ == "__main__":
    unittest.main()
