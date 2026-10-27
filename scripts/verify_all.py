"""
Unified multi-script verification runner for MIMIC-IV Ventilator System.
Sequentially runs all team verification scripts (verify_setup.py, verify_data_paths.py, verify_web_assets.py)
and reports an integrated health check summary.
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

VERIFICATION_SCRIPTS = [
    ("verify_setup.py", "Environment & Dependencies Sanity Check"),
    ("verify_data_paths.py", "Dataset Path & Clinical Tables Verification"),
    ("verify_web_assets.py", "Web Frontend Assets & Server Boot Verification"),
]


def run_verification_suite() -> bool:
    print("=" * 65)
    print("MIMIC-IV VENTILATOR SYSTEM - FULL VERIFICATION SUITE")
    print("=" * 65)
    print(f"Executing {len(VERIFICATION_SCRIPTS)} verification test suites locally...\n")

    results = []
    all_passed = True

    for idx, (script_name, description) in enumerate(VERIFICATION_SCRIPTS, 1):
        script_path = SCRIPTS_DIR / script_name
        print(f"\n[{idx}/{len(VERIFICATION_SCRIPTS)}] Running: {script_name} ({description})")
        print("-" * 65)

        if not script_path.is_file():
            print(f"[FAIL] Script not found: {script_path}")
            results.append((script_name, description, False))
            all_passed = False
            continue

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                check=False
            )
            passed = (proc.returncode == 0)
            results.append((script_name, description, passed))
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"[ERROR] Exception running {script_name}: {e}")
            results.append((script_name, description, False))
            all_passed = False

    print("\n" + "=" * 65)
    print("MULTI-SCRIPT VERIFICATION SUMMARY")
    print("=" * 65)
    for script_name, description, passed in results:
        status_tag = "[PASS]" if passed else "[FAIL]"
        print(f"  {status_tag:<8} {script_name:<24} - {description}")

    print("=" * 65)
    if all_passed:
        print("[SUCCESS] All verification scripts completed successfully with code 0!")
        return True
    else:
        print("[ERROR] One or more verification suites failed. Review logs above.")
        return False


def main():
    success = run_verification_suite()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
