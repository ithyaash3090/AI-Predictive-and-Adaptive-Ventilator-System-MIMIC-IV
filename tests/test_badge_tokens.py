"""
Unit tests for clinical telemetry status badge tokens and contrast compliance.
Verifies Task 1.10 requirements in src/web/static/styles.css and dashboard templates.
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STYLES_PATH = PROJECT_ROOT / "src" / "web" / "static" / "styles.css"
TEMPLATE_PATH = PROJECT_ROOT / "src" / "web" / "templates" / "dashboard.html"


def test_badge_tokens_in_root():
    """Verifies that all required clinical status badge design tokens are declared in :root."""
    css = STYLES_PATH.read_text(encoding="utf-8")
    required_tokens = [
        "--badge-normal-bg",
        "--badge-normal-border",
        "--badge-normal-text",
        "--badge-warning-bg",
        "--badge-warning-border",
        "--badge-warning-text",
        "--badge-critical-bg",
        "--badge-critical-border",
        "--badge-critical-text",
        "--badge-supervised-bg",
        "--badge-supervised-border",
        "--badge-supervised-text",
    ]
    for token in required_tokens:
        assert token in css, f"Missing required design token: {token}"


def test_badge_classes_defined():
    """Verifies that .badge-normal, .badge-warning, .badge-critical, .badge-supervised exist."""
    css = STYLES_PATH.read_text(encoding="utf-8")
    required_classes = [
        ".badge-normal",
        ".badge-warning",
        ".badge-critical",
        ".badge-supervised",
    ]
    for cls in required_classes:
        assert cls in css, f"Missing badge class: {cls}"


def test_micro_transitions_and_animations():
    """Verifies that micro-transitions and pulse animations are defined."""
    css = STYLES_PATH.read_text(encoding="utf-8")
    assert "pulseCritical" in css, "Missing pulseCritical animation"
    assert "pulseWarning" in css, "Missing pulseWarning animation"
    assert "transition:" in css or "transition :" in css, "Missing micro-transition styling"


def test_telemetry_tile_integration():
    """Verifies that telemetry cards integrate with badge tokens and tile headers."""
    css = STYLES_PATH.read_text(encoding="utf-8")
    html = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert ".tile-header" in css, "Missing .tile-header class in styles.css"
    assert "state-normal" in css, "Missing state-normal class in styles.css"
    assert "state-critical" in css, "Missing state-critical class in styles.css"

    assert "badge-status-spo2" in html, "Missing SpO2 status badge in dashboard.html"
    assert "badge-status-dp" in html, "Missing Driving Pressure status badge in dashboard.html"
    assert "badge-normal" in html, "Missing badge-normal class in dashboard.html"


if __name__ == "__main__":
    test_badge_tokens_in_root()
    test_badge_classes_defined()
    test_micro_transitions_and_animations()
    test_telemetry_tile_integration()
    print("[PASS] All telemetry status badge token unit tests passed successfully!")
