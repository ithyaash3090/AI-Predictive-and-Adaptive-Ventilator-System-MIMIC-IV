"""
Dataset Verification and Setup Helper for MIMIC-IV Clinical Database Demo v2.2.
Verifies that all required hosp/ and icu/ tables are present and ready for extraction.
"""

import os
import sys

DATASET_ROOT = "mimic-iv-demo-dataset/mimic-iv-clinical-database-demo-2.2"
REQUIRED_TABLES = [
    ("icu", "chartevents"),
    ("icu", "icustays"),
    ("icu", "d_items"),
    ("icu", "procedureevents"),
    ("hosp", "patients"),
    ("hosp", "admissions"),
    ("hosp", "diagnoses_icd"),
    ("hosp", "d_icd_diagnoses"),
    ("hosp", "labevents"),
    ("hosp", "d_labitems")
]

def verify_dataset():
    print("="*65)
    print("      MIMIC-IV DEMO DATASET INTEGRITY & ACCESSIBILITY CHECK")
    print("="*65)

    if not os.path.exists(DATASET_ROOT):
        print(f"\n[ERROR] Dataset directory not found at: {DATASET_ROOT}")
        print("\nTo obtain the MIMIC-IV Clinical Database Demo (v2.2):")
        print("  1. Visit: https://physionet.org/content/mimic-iv-demo/2.2/")
        print("  2. Download the demo zip archive (open access, no credentialing needed).")
        print(f"  3. Extract contents into: {DATASET_ROOT}")
        return False

    all_present = True
    print("\nChecking required clinical tables:")
    for folder, table_name in REQUIRED_TABLES:
        # Check both flat and nested folder paths
        p1 = os.path.join(DATASET_ROOT, folder, f"{table_name}.csv", f"{table_name}.csv")
        p2 = os.path.join(DATASET_ROOT, folder, f"{table_name}.csv")
        
        if os.path.exists(p1):
            sz_mb = os.path.getsize(p1) / (1024 * 1024)
            print(f"  [OK] {folder}/{table_name}.csv ({sz_mb:.2f} MB)")
        elif os.path.exists(p2):
            sz_mb = os.path.getsize(p2) / (1024 * 1024)
            print(f"  [OK] {folder}/{table_name}.csv ({sz_mb:.2f} MB)")
        else:
            print(f"  [MISSING] {folder}/{table_name}.csv")
            all_present = False

    print("-" * 65)
    if all_present:
        print("[SUCCESS] All required MIMIC-IV demo tables are present and verified!")
        print("You can proceed to run:")
        print("  python scripts/build_dataset.py   # Extracts cohort time-series")
        print("  python scripts/train_models.py    # Trains and benchmarks models")
        print("  python -m uvicorn src.web.app:app # Starts clinical dashboard")
        return True
    else:
        print("[WARNING] Some clinical tables were not found. Check extraction folder.")
        return False

if __name__ == "__main__":
    success = verify_dataset()
    sys.exit(0 if success else 1)
