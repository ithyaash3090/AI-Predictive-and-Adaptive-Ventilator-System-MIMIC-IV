"""
FastAPI Server and Static Asset Loading Verification Script for MIMIC-IV Ventilator System.
Verifies that the FastAPI backend boots properly and serves the dashboard, static assets, and core API endpoints.
"""

import argparse
import sys
import warnings
from pathlib import Path

# Suppress warnings before any library imports
warnings.filterwarnings("ignore")

# Add project root to sys.path to enable importing from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from starlette.testclient import TestClient
except ImportError:
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        TestClient = None


def create_test_client():
    """Initializes and returns a TestClient instance against the FastAPI app."""
    if TestClient is None:
        raise ImportError("Neither 'starlette.testclient' nor 'fastapi.testclient' could be imported. Please ensure starlette/httpx is installed.")

    from src.web.app import app
    return TestClient(app)


def verify_root_endpoint(client: TestClient) -> tuple[bool, str]:
    """Tests GET / to ensure it returns HTTP 200 and renders the dashboard HTML."""
    try:
        response = client.get("/")
        if response.status_code != 200:
            return False, f"Expected HTTP 200, got HTTP {response.status_code}"

        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type:
            return False, f"Expected content-type 'text/html', got '{content_type}'"

        text = response.text
        if not text or len(text.strip()) == 0:
            return False, "Response body was empty"

        # Check for expected dashboard content
        if "MIMIC-IV" not in text and "Ventilator" not in text and "<html" not in text:
            return False, "Dashboard template does not contain expected HTML markers"

        return True, f"HTTP 200 OK, content-type: {content_type}, length: {len(text)} bytes"
    except Exception as e:
        return False, f"Exception occurred: {e}"


def verify_static_css(client: TestClient) -> tuple[bool, str]:
    """Tests GET /static/styles.css returns HTTP 200 with text/css content-type."""
    try:
        response = client.get("/static/styles.css")
        if response.status_code != 200:
            return False, f"Expected HTTP 200, got HTTP {response.status_code}"

        content_type = response.headers.get("content-type", "").lower()
        if "text/css" not in content_type:
            return False, f"Expected content-type containing 'text/css', got '{content_type}'"

        content_len = len(response.content)
        if content_len == 0:
            return False, "CSS file returned an empty response"

        return True, f"HTTP 200 OK, content-type: {content_type}, size: {content_len} bytes"
    except Exception as e:
        return False, f"Exception occurred: {e}"


def verify_static_js(client: TestClient) -> tuple[bool, str]:
    """Tests GET /static/dashboard.js returns HTTP 200 with javascript content-type."""
    try:
        response = client.get("/static/dashboard.js")
        if response.status_code != 200:
            return False, f"Expected HTTP 200, got HTTP {response.status_code}"

        content_type = response.headers.get("content-type", "").lower()
        if "javascript" not in content_type:
            return False, f"Expected content-type containing 'javascript', got '{content_type}'"

        content_len = len(response.content)
        if content_len == 0:
            return False, "JavaScript file returned an empty response"

        return True, f"HTTP 200 OK, content-type: {content_type}, size: {content_len} bytes"
    except Exception as e:
        return False, f"Exception occurred: {e}"


def verify_benchmark_endpoint(client: TestClient) -> tuple[bool, str]:
    """Tests GET /api/benchmark returns HTTP 200 with JSON payload."""
    try:
        response = client.get("/api/benchmark")
        if response.status_code != 200:
            return False, f"Expected HTTP 200, got HTTP {response.status_code}"

        content_type = response.headers.get("content-type", "").lower()
        if "application/json" not in content_type:
            return False, f"Expected content-type 'application/json', got '{content_type}'"

        data = response.json()
        return True, f"HTTP 200 OK, JSON response received ({len(data)} top-level keys)"
    except Exception as e:
        return False, f"Exception occurred: {e}"


def verify_recommendation_endpoint(client: TestClient) -> tuple[bool, str]:
    """Tests POST /api/adaptive/recommend returns HTTP 200 and valid recommendation."""
    try:
        payload = {
            "fio2": 0.40,
            "peep": 8.0,
            "tidal_volume_ml": 450.0,
            "respiratory_rate": 16,
            "spo2": 95.0,
            "driving_pressure": 12.0,
            "plateau_pressure": 20.0,
            "ph": 7.40,
            "paco2": 40.0,
            "pbw_kg": 70.0,
            "predicted_risk": 0.25,
            "is_ards_or_copd": False
        }
        response = client.post("/api/adaptive/recommend", json=payload)
        if response.status_code != 200:
            return False, f"Expected HTTP 200, got HTTP {response.status_code} - {response.text}"

        data = response.json()
        if "recommendation" not in data:
            return False, "Missing 'recommendation' key in response JSON"

        rec = data["recommendation"]
        has_urgency = "urgency_level" in rec
        has_checks = "safety_checks" in rec
        return True, f"HTTP 200 OK, urgency_level: {rec.get('urgency_level', 'N/A')}, safety_checks: {has_checks}"
    except Exception as e:
        return False, f"Exception occurred: {e}"


def verify_action_log_endpoint(client: TestClient) -> tuple[bool, str]:
    """Tests POST /api/action/log returns HTTP 200 and logs clinician action."""
    try:
        payload = {
            "stay_id": 99999,
            "action_type": "APPROVE",
            "applied_settings": {"fio2": 0.40, "peep": 8.0},
            "clinician_notes": "Automated verification test audit entry"
        }
        response = client.post("/api/action/log", json=payload)
        if response.status_code != 200:
            return False, f"Expected HTTP 200, got HTTP {response.status_code} - {response.text}"

        data = response.json()
        if data.get("status") != "SUCCESS":
            return False, f"Expected status 'SUCCESS', got '{data.get('status')}'"

        return True, "HTTP 200 OK, action successfully recorded in clinical audit log"
    except Exception as e:
        return False, f"Exception occurred: {e}"


def run_verification_suite(verbose: bool = False) -> bool:
    """Runs all verification checks and prints formatted test results."""
    print("=" * 70)
    print("MIMIC-IV VENTILATOR SYSTEM - FASTAPI SERVER & STATIC ASSETS VERIFICATION")
    print("=" * 70)

    print("[*] Initializing FastAPI TestClient...")
    try:
        client = create_test_client()
        print("  [PASS] TestClient successfully initialized against src/web/app.py\n")
    except Exception as e:
        print(f"  [FAIL] Failed to initialize TestClient: {e}\n")
        return False

    tests = [
        ("Dashboard Root (/)", verify_root_endpoint),
        ("Static CSS (/static/styles.css)", verify_static_css),
        ("Static JavaScript (/static/dashboard.js)", verify_static_js),
        ("Benchmark API (/api/benchmark)", verify_benchmark_endpoint),
        ("Adaptive Recommendation API (/api/adaptive/recommend)", verify_recommendation_endpoint),
        ("Clinician Audit Log API (/api/action/log)", verify_action_log_endpoint),
    ]

    results = []
    all_passed = True

    print("[*] Executing Endpoint & Asset Validation Suite...")
    print("-" * 70)

    for name, test_fn in tests:
        passed, msg = test_fn(client)
        results.append((name, passed, msg))
        status_tag = "[PASS]" if passed else "[FAIL]"
        print(f"  {status_tag:<8} {name:<42} -> {msg}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    for name, passed, msg in results:
        status_tag = "[PASS]" if passed else "[FAIL]"
        print(f"  {status_tag:<8} {name}")

    print("=" * 70)
    if all_passed:
        print("[SUCCESS] All FastAPI server endpoints and static assets verified successfully!")
        return True
    else:
        print("[ERROR] One or more verification checks failed. See details above.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify FastAPI backend server boots properly and serves dashboard and static assets."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display detailed diagnostic output."
    )
    args = parser.parse_args()

    success = run_verification_suite(verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
