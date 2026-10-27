"""
Unit tests for MIMIC-IV Ventilator web assets and FastAPI server verification.
Validates existence of templates, static CSS/JS assets, FastAPI server initialization, and routes.
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

from verify_web_assets import (
    REQUIRED_STATIC_ASSETS,
    REQUIRED_TEMPLATES,
    check_fastapi_app,
    check_static_assets,
    check_templates,
    format_file_size,
)


class TestVerifyWebAssets(unittest.TestCase):
    def test_required_templates_exist(self):
        """Verifies that all required HTML templates exist and have non-zero size."""
        self.assertTrue(check_templates(PROJECT_ROOT))
        for rel_template in REQUIRED_TEMPLATES:
            template_path = PROJECT_ROOT / rel_template
            self.assertTrue(template_path.is_file(), f"Template missing: {rel_template}")
            self.assertGreater(template_path.stat().st_size, 0, f"Template is empty: {rel_template}")

    def test_required_static_assets_exist(self):
        """Verifies that static stylesheet and JavaScript files exist and have non-zero size."""
        self.assertTrue(check_static_assets(PROJECT_ROOT))
        for rel_asset in REQUIRED_STATIC_ASSETS:
            asset_path = PROJECT_ROOT / rel_asset
            self.assertTrue(asset_path.is_file(), f"Static asset missing: {rel_asset}")
            self.assertGreater(asset_path.stat().st_size, 0, f"Static asset is empty: {rel_asset}")

    def test_fastapi_app_initialization(self):
        """Verifies that FastAPI application initializes with correct title and routes."""
        self.assertTrue(check_fastapi_app(check_routes=True))

    def test_format_file_size(self):
        """Tests file size human-readable formatting utility."""
        self.assertEqual(format_file_size(512), "0.5 KB")
        self.assertEqual(format_file_size(2048), "2.0 KB")
        self.assertEqual(format_file_size(2 * 1024 * 1024), "2.0 MB (2048.0 KB)")


if __name__ == "__main__":
    unittest.main()
