"""
Unit tests for clinical deterioration threshold labeling logic and lookahead alignment.
Validates Berlin ARDS criteria, ROX index thresholds, multi-criteria deterioration flags,
and strict avoidance of lookahead bias in src/data/extractor.py.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.extractor import MimicIvExtractor


class TestDeteriorationLabeling(unittest.TestCase):
    """Test suite for clinical deterioration labeling and lookahead window construction."""

    def setUp(self):
        self.extractor = MimicIvExtractor()

    def test_instant_deterioration_criteria_triggers(self):
        """Verifies that each individual physiological condition triggers instant deterioration."""
        # Baseline normal values
        base_record = {
            "stay_id": 1,
            "hour_from_intime": 0,
            "sf_ratio": 300.0,
            "pf_ratio": 350.0,
            "fio2": 0.30,
            "spo2": 98.0,
            "driving_pressure": 10.0,
            "ph": 7.40,
            "paco2": 40.0,
            "respiratory_rate": 16.0,
            "rox_index": 12.0,
        }

        # 1. Hypoxemia (SF < 180 or PF <= 200)
        rec_hypoxemia = base_record.copy()
        rec_hypoxemia["sf_ratio"] = 175.0
        df = pd.DataFrame([base_record, rec_hypoxemia])
        df_out = self.extractor.label_respiratory_deterioration(df, lookahead_hours=1)
        self.assertEqual(df_out.iloc[0]["instant_deterioration"], 0)
        self.assertEqual(df_out.iloc[1]["instant_deterioration"], 1)
        self.assertEqual(df_out.iloc[1]["flag_hypoxemia"], 1)

        # 2. Refractory Desaturation (FiO2 >= 0.70 & SpO2 < 90)
        rec_support = base_record.copy()
        rec_support["fio2"] = 0.80
        rec_support["spo2"] = 88.0
        df = pd.DataFrame([rec_support])
        df_out = self.extractor.label_respiratory_deterioration(df, lookahead_hours=1)
        self.assertEqual(df_out.iloc[0]["instant_deterioration"], 1)
        self.assertEqual(df_out.iloc[0]["flag_high_support"], 1)

        # 3. High Driving Pressure (> 18.0)
        rec_dp = base_record.copy()
        rec_dp["driving_pressure"] = 19.5
        df = pd.DataFrame([rec_dp])
        df_out = self.extractor.label_respiratory_deterioration(df, lookahead_hours=1)
        self.assertEqual(df_out.iloc[0]["instant_deterioration"], 1)
        self.assertEqual(df_out.iloc[0]["flag_high_driving_pressure"], 1)

        # 4. Acute Respiratory Acidosis (pH < 7.25 & PaCO2 > 50)
        rec_acidosis = base_record.copy()
        rec_acidosis["ph"] = 7.20
        rec_acidosis["paco2"] = 55.0
        df = pd.DataFrame([rec_acidosis])
        df_out = self.extractor.label_respiratory_deterioration(df, lookahead_hours=1)
        self.assertEqual(df_out.iloc[0]["instant_deterioration"], 1)
        self.assertEqual(df_out.iloc[0]["flag_respiratory_acidosis"], 1)

        # 5. Tachypneic Exhaustion (RR >= 32 & ROX < 4.88)
        rec_exhaustion = base_record.copy()
        rec_exhaustion["respiratory_rate"] = 35.0
        rec_exhaustion["rox_index"] = 4.2
        df = pd.DataFrame([rec_exhaustion])
        df_out = self.extractor.label_respiratory_deterioration(df, lookahead_hours=1)
        self.assertEqual(df_out.iloc[0]["instant_deterioration"], 1)
        self.assertEqual(df_out.iloc[0]["flag_respiratory_exhaustion"], 1)

    def test_strict_lookahead_timing_and_no_self_leakage(self):
        """
        Verifies that instant deterioration at hour t does NOT set target at hour t to 1,
        and that target at hour t correctly captures deterioration in [t+1, t+4].
        """
        hours = 8
        records = []
        for h in range(hours):
            records.append({
                "stay_id": 100,
                "hour_from_intime": h,
                "sf_ratio": 300.0 if h != 4 else 150.0,  # Deterioration only at hour 4
                "pf_ratio": 350.0 if h != 4 else 140.0,
                "fio2": 0.30,
                "spo2": 98.0,
                "driving_pressure": 10.0,
                "ph": 7.40,
                "paco2": 40.0,
                "respiratory_rate": 16.0,
                "rox_index": 12.0,
            })

        df = pd.DataFrame(records)
        df_out = self.extractor.label_respiratory_deterioration(df, lookahead_hours=4)

        # Instant deterioration should occur ONLY at hour 4
        self.assertEqual(list(df_out["instant_deterioration"]), [0, 0, 0, 0, 1, 0, 0, 0])

        # Forward target window of size 4:
        # Hour 0: looks at h1..h4 -> h4 is 1 -> target is 1
        # Hour 1: looks at h2..h5 -> h4 is 1 -> target is 1
        # Hour 2: looks at h3..h6 -> h4 is 1 -> target is 1
        # Hour 3: looks at h4..h7 -> h4 is 1 -> target is 1
        # Hour 4: looks at h5..h8 -> no events in h5..h7 -> target MUST BE 0 (no self-leakage!)
        # Hour 5: looks at h6..h8 -> no events -> 0
        # Hour 6: looks at h7..h8 -> no events -> 0
        # Hour 7: looks at h8 -> no events -> 0
        expected_forward_target = [1, 1, 1, 1, 0, 0, 0, 0]
        self.assertEqual(list(df_out["target_deterioration_lead"]), expected_forward_target)
        self.assertEqual(list(df_out["deterioration_next_4h"]), expected_forward_target)

    def test_multi_stay_isolation(self):
        """Verifies that forward windows do not bleed across different ICU stays."""
        # Stay 1 ends at h2 with no deterioration
        # Stay 2 starts at h0 with instant deterioration at h1
        records = [
            {"stay_id": 1, "hour_from_intime": 0, "sf_ratio": 300.0, "pf_ratio": 350.0, "fio2": 0.3, "spo2": 98.0, "driving_pressure": 10.0, "ph": 7.4, "paco2": 40.0, "respiratory_rate": 16.0, "rox_index": 12.0},
            {"stay_id": 1, "hour_from_intime": 1, "sf_ratio": 300.0, "pf_ratio": 350.0, "fio2": 0.3, "spo2": 98.0, "driving_pressure": 10.0, "ph": 7.4, "paco2": 40.0, "respiratory_rate": 16.0, "rox_index": 12.0},
            # Stay 2
            {"stay_id": 2, "hour_from_intime": 0, "sf_ratio": 300.0, "pf_ratio": 350.0, "fio2": 0.3, "spo2": 98.0, "driving_pressure": 10.0, "ph": 7.4, "paco2": 40.0, "respiratory_rate": 16.0, "rox_index": 12.0},
            {"stay_id": 2, "hour_from_intime": 1, "sf_ratio": 150.0, "pf_ratio": 130.0, "fio2": 0.3, "spo2": 98.0, "driving_pressure": 10.0, "ph": 7.4, "paco2": 40.0, "respiratory_rate": 16.0, "rox_index": 12.0},
        ]
        df = pd.DataFrame(records)
        df_out = self.extractor.label_respiratory_deterioration(df, lookahead_hours=2)

        # Stay 1 records should both have target 0 (no bleed from Stay 2)
        stay1_targets = list(df_out[df_out["stay_id"] == 1]["target_deterioration_lead"])
        self.assertEqual(stay1_targets, [0, 0])

        # Stay 2 h0 looks at h1 -> target is 1
        stay2_targets = list(df_out[df_out["stay_id"] == 2]["target_deterioration_lead"])
        self.assertEqual(stay2_targets, [1, 0])

    def test_audit_label_distribution_summary(self):
        """Verifies summary statistics generated by audit_label_distribution."""
        records = [
            {"stay_id": 1, "subject_id": 101, "hour_from_intime": 0, "sf_ratio": 300.0, "pf_ratio": 350.0, "fio2": 0.3, "spo2": 98.0, "driving_pressure": 10.0, "ph": 7.4, "paco2": 40.0, "respiratory_rate": 16.0, "rox_index": 12.0},
            {"stay_id": 1, "subject_id": 101, "hour_from_intime": 1, "sf_ratio": 150.0, "pf_ratio": 140.0, "fio2": 0.3, "spo2": 98.0, "driving_pressure": 10.0, "ph": 7.4, "paco2": 40.0, "respiratory_rate": 16.0, "rox_index": 12.0},
        ]
        df = pd.DataFrame(records)
        df_out = self.extractor.label_respiratory_deterioration(df, lookahead_hours=1)
        audit = MimicIvExtractor.audit_label_distribution(df_out)

        self.assertEqual(audit["total_rows"], 2)
        self.assertEqual(audit["total_stays"], 1)
        self.assertEqual(audit["total_patients"], 1)
        self.assertEqual(audit["positive_count"], 1)
        self.assertEqual(audit["negative_count"], 1)
        self.assertEqual(audit["criteria_breakdown"]["hypoxemia"], 1)


if __name__ == "__main__":
    unittest.main()
