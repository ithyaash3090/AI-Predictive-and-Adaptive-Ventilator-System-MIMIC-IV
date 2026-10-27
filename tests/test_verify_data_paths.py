"""
Unit tests for MIMIC-IV dataset path and structure verification.
Validates table resolution, size formatting, dataset path presence, and table integrity.
"""

import sys
import unittest
from pathlib import Path

# Add project root and scripts to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from verify_data_paths import (
    REL_DATASET_DIR,
    REQUIRED_SUBDIRS,
    REQUIRED_TABLES,
    check_table_integrity,
    format_file_size,
    resolve_table_path,
    verify_dataset,
)


class TestVerifyDataPaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = PROJECT_ROOT / REL_DATASET_DIR

    def test_base_dataset_directory_exists(self):
        """Verifies that the MIMIC-IV demo dataset root directory exists."""
        self.assertTrue(
            self.base_dir.is_dir(),
            f"Expected dataset root to exist at {self.base_dir}",
        )

    def test_required_subdirectories_exist(self):
        """Verifies that hosp/ and icu/ subdirectories exist."""
        for subdir in REQUIRED_SUBDIRS:
            sub_path = self.base_dir / subdir
            self.assertTrue(
                sub_path.is_dir(),
                f"Missing required subdirectory: {subdir}",
            )

    def test_required_tables_resolve_successfully(self):
        """Verifies each required clinical table resolves to an existing file."""
        for rel_table in REQUIRED_TABLES:
            resolved = resolve_table_path(self.base_dir, rel_table)
            self.assertIsNotNone(
                resolved,
                f"Failed to resolve table: {rel_table}",
            )
            self.assertTrue(
                resolved.is_file(),
                f"Resolved path is not a file: {resolved}",
            )
            self.assertGreater(
                resolved.stat().st_size,
                0,
                f"Table file is empty: {resolved}",
            )

    def test_resolve_missing_table_returns_none(self):
        """Verifies that non-existent table path returns None."""
        missing = resolve_table_path(self.base_dir, "hosp/non_existent_table.csv.gz")
        self.assertIsNone(missing)

    def test_format_file_size(self):
        """Tests file size human-readable formatting."""
        self.assertEqual(format_file_size(512), "0.5 KB")
        self.assertEqual(format_file_size(1024), "1.0 KB")
        self.assertEqual(format_file_size(1024 * 1024), "1.0 MB (1024.0 KB)")

    def test_check_table_integrity(self):
        """Verifies that each resolved table can be parsed and has valid columns."""
        for rel_table in REQUIRED_TABLES:
            resolved = resolve_table_path(self.base_dir, rel_table)
            self.assertIsNotNone(resolved)
            ok, info = check_table_integrity(resolved)
            self.assertTrue(ok, f"Integrity check failed for {rel_table}: {info}")
            self.assertIn("columns verified", info)

    def test_verify_dataset_suite(self):
        """Tests that full dataset verification passes with code 0."""
        self.assertTrue(verify_dataset(check_integrity=True))


if __name__ == "__main__":
    unittest.main()
