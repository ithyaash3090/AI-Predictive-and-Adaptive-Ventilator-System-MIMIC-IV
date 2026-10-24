"""
Environment verification and sanity check script for MIMIC-IV Ventilator System.
Checks Python version, core dependencies, and demo dataset presence.
"""

import sys
from pathlib import Path

REQUIRED_PYTHON = (3, 10)
REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scikit-learn", "sklearn"),
    ("xgboost", "xgboost"),
    ("lightgbm", "lightgbm"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
]


def check_python_version() -> bool:
    current = sys.version_info[:2]
    ver_str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if current >= REQUIRED_PYTHON:
        print(f"[PASS] Python version: {ver_str} (>= {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]})")
        return True
    print(f"[FAIL] Python version: {ver_str} (Requires Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} or higher)")
    return False


def check_dependencies() -> bool:
    all_ok = True
    print("[*] Checking core dependencies...")
    for pkg_name, import_name in REQUIRED_PACKAGES:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [PASS] {pkg_name} ({ver}) imported successfully")
        except ImportError as e:
            print(f"  [FAIL] Failed to import {pkg_name}: {e}")
            all_ok = False

    if all_ok:
        print("[OK] All core dependencies imported successfully")
    else:
        print("[FAIL] Some dependencies are missing. Run: pip install -r requirements.txt")
    return all_ok


def check_dataset_directory() -> bool:
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "mimic-iv-demo-dataset"

    if dataset_dir.is_dir():
        print(f"[PASS] Demo dataset directory found: {dataset_dir.name}/")
        return True
    print(f"[FAIL] Demo dataset directory NOT found at: {dataset_dir}")
    return False


def main():
    print("=" * 60)
    print("AI-Predictive-and-Adaptive-Ventilator-System Sanity Check")
    print("=" * 60)

    py_ok = check_python_version()
    print("-" * 60)

    pkg_ok = check_dependencies()
    print("-" * 60)

    data_ok = check_dataset_directory()
    print("=" * 60)

    if py_ok and pkg_ok and data_ok:
        print("[SUCCESS] Environment verification PASSED! Ready for development.")
        sys.exit(0)
    else:
        print("[ERROR] Environment verification FAILED. Please resolve errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

