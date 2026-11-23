"""
Unit tests for clinical vital signs distribution histograms & KDE visualization.
Verifies scripts/plot_vital_distributions.py functions:
- compute_vital_statistics
- compute_all_vital_statistics
- format_terminal_summary
- plot_vital_distributions
- end-to-end pipeline execution
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root and scripts directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from plot_vital_distributions import (
    VITALS_CONFIG,
    compute_all_vital_statistics,
    compute_vital_statistics,
    format_terminal_summary,
    plot_vital_distributions,
    run_vitals_visualization,
)


class TestPlotVitalDistributions(unittest.TestCase):
    """Test suite for vital signs statistical calculation, reporting, and plotting."""

    def setUp(self):
        # Known synthetic distribution
        np.random.seed(42)
        self.synthetic_hr = pd.Series(np.linspace(60, 100, 41))  # 60, 61, ..., 100
        self.synthetic_vitals = {
            "heart_rate": self.synthetic_hr,
            "respiratory_rate": pd.Series(np.linspace(12, 28, 33)),
            "spo2": pd.Series(np.linspace(90, 100, 21)),
            "systolic_bp": pd.Series(np.linspace(90, 150, 31)),
            "diastolic_bp": pd.Series(np.linspace(50, 90, 21)),
            "temperature": pd.Series(np.linspace(36.0, 38.5, 26)),
        }

    def test_compute_vital_statistics(self):
        """Verifies mean, median, percentiles, std, and IQR on known series."""
        stats = compute_vital_statistics(self.synthetic_hr)
        self.assertEqual(stats["count"], 41)
        self.assertAlmostEqual(stats["mean"], 80.0, places=1)
        self.assertAlmostEqual(stats["median"], 80.0, places=1)
        self.assertAlmostEqual(stats["min"], 60.0, places=1)
        self.assertAlmostEqual(stats["max"], 100.0, places=1)
        self.assertAlmostEqual(stats["iqr"], 20.0, places=1)
        self.assertAlmostEqual(stats["p5"], 62.0, places=1)
        self.assertAlmostEqual(stats["p95"], 98.0, places=1)

    def test_compute_vital_statistics_empty(self):
        """Verifies safe handling of empty series without exceptions."""
        empty_stats = compute_vital_statistics(pd.Series([], dtype=float))
        self.assertEqual(empty_stats["count"], 0)
        self.assertEqual(empty_stats["mean"], 0.0)
        self.assertEqual(empty_stats["median"], 0.0)

    def test_compute_all_vital_statistics(self):
        """Verifies dictionary creation across all vital sign keys."""
        all_stats = compute_all_vital_statistics(self.synthetic_vitals)
        for cfg in VITALS_CONFIG:
            k = cfg["key"]
            self.assertIn(k, all_stats)
            self.assertGreater(all_stats[k]["count"], 0)

    def test_format_terminal_summary(self):
        """Verifies formatted terminal summary string contents."""
        all_stats = compute_all_vital_statistics(self.synthetic_vitals)
        text = format_terminal_summary(all_stats, source_desc="Unit Test Synthetic Data")
        self.assertIn("MIMIC-IV CLINICAL VITAL SIGNS DISTRIBUTION SUMMARY", text)
        self.assertIn("Heart Rate", text)
        self.assertIn("Respiratory Rate", text)
        self.assertIn("Oxygen Saturation (SpO2)", text)
        self.assertIn("Systolic Blood Pressure", text)
        self.assertIn("Diastolic Blood Pressure", text)
        self.assertIn("Body Temperature", text)

    def test_plot_vital_distributions_synthetic(self):
        """Verifies that plot_vital_distributions produces a valid non-empty PNG file."""
        all_stats = compute_all_vital_statistics(self.synthetic_vitals)
        test_output = PROJECT_ROOT / "outputs" / "test_vital_plot.png"

        try:
            res_path = plot_vital_distributions(
                self.synthetic_vitals,
                all_stats,
                output_path=test_output,
                dpi=100,
            )
            self.assertTrue(test_output.is_file())
            self.assertGreater(test_output.stat().st_size, 1000)  # > 1KB
            self.assertEqual(res_path, test_output)
        finally:
            if test_output.is_file():
                test_output.unlink()

    def test_pipeline_execution_on_cohort(self):
        """Verifies end-to-end pipeline execution and output artifact creation."""
        test_out = PROJECT_ROOT / "outputs" / "test_pipeline_vital_distributions.png"
        try:
            success = run_vitals_visualization(
                output_png=test_out,
                dpi=100,
                quiet=True,
            )
            self.assertTrue(success)
            self.assertTrue(test_out.is_file())
            self.assertGreater(test_out.stat().st_size, 5000)
        finally:
            if test_out.is_file():
                test_out.unlink()


if __name__ == "__main__":
    unittest.main()
