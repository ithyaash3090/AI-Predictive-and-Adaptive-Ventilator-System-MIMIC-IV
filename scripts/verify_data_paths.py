"""
Dataset path and structure verification script for MIMIC-IV Ventilator System.
Verifies existence of dataset directories and core clinical table files.
"""

import argparse
import csv
import gzip
import sys
from pathlib import Path

# Relative dataset path from project root
REL_DATASET_DIR = Path("mimic-iv-demo-dataset") / "mimic-iv-clinical-database-demo-2.2"
REQUIRED_SUBDIRS = ["hosp", "icu"]
REQUIRED_TABLES = [
    "hosp/patients.csv.gz",
    "hosp/admissions.csv.gz",
    "hosp/diagnoses_icd.csv.gz",
    "hosp/labevents.csv.gz",
    "icu/icustays.csv.gz",
    "icu/chartevents.csv.gz",
]


def resolve_table_path(base_dir: Path, rel_table_path: str):
    """
    Resolves the table file across standard distribution formats:
    - Direct compressed: base_dir / rel_table_path (e.g. hosp/patients.csv.gz)
    - Uncompressed flat: base_dir / rel_table_path.csv (e.g. hosp/patients.csv)
    - Git demo nested: base_dir / folder / stem.csv / stem.csv (e.g. hosp/patients.csv/patients.csv)
    """
    direct = base_dir / rel_table_path
    if direct.is_file():
        return direct

    # Check flat uncompressed (.csv)
    flat_csv = base_dir / rel_table_path.replace(".csv.gz", ".csv")
    if flat_csv.is_file():
        return flat_csv

    # Check nested directory structure common in git checkouts
    folder, fname = rel_table_path.split("/")
    stem = fname.replace(".csv.gz", "").replace(".csv", "")
    nested_csv = base_dir / folder / f"{stem}.csv" / f"{stem}.csv"
    if nested_csv.is_file():
        return nested_csv

    return None


def format_file_size(size_bytes: int) -> str:
    """Formats file size into human-readable KB or MB."""
    size_kb = size_bytes / 1024
    if size_kb >= 1024:
        return f"{size_kb / 1024:.1f} MB ({size_kb:.1f} KB)"
    return f"{size_kb:.1f} KB"


def check_table_integrity(resolved_path: Path) -> tuple[bool, str]:
    """
    Validates that a resolved table file can be opened and contains a valid header row.
    Supports both gzip-compressed (.csv.gz) and standard (.csv) files.
    """
    try:
        if resolved_path.name.endswith(".gz"):
            with gzip.open(resolved_path, "rt", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, None)
        else:
            with open(resolved_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, None)

        if not header:
            return False, "Empty table or missing header row"
        return True, f"{len(header)} columns verified"
    except Exception as e:
        return False, str(e)


def verify_dataset(check_integrity: bool = False) -> bool:
    project_root = Path(__file__).resolve().parent.parent
    base_dir = project_root / REL_DATASET_DIR

    all_passed = True

    print("=" * 65)
    print("MIMIC-IV Clinical Database Demo Path Verification")
    print("=" * 65)

    # 1. Check base dataset directory
    print("[*] Checking base dataset directory...")
    if base_dir.is_dir():
        print(f"[PASS] Dataset root found: {REL_DATASET_DIR.as_posix()}/")
    else:
        print(f"[FAIL] Dataset root NOT found at: {base_dir}")
        return False

    print("-" * 65)

    # 2. Check hosp/ and icu/ subdirectories
    print("[*] Checking required clinical subdirectories...")
    for subdir_name in REQUIRED_SUBDIRS:
        sub_path = base_dir / subdir_name
        if sub_path.is_dir():
            print(f"[PASS] Subdirectory found: {subdir_name}/")
        else:
            print(f"[FAIL] Missing required subdirectory: {subdir_name}/")
            all_passed = False

    print("-" * 65)

    # 3. Check essential table files
    print("[*] Checking essential clinical table files...")
    for table_rel in REQUIRED_TABLES:
        resolved_file = resolve_table_path(base_dir, table_rel)
        if resolved_file:
            size_str = format_file_size(resolved_file.stat().st_size)
            if check_integrity:
                ok, integrity_info = check_table_integrity(resolved_file)
                if ok:
                    print(f"[PASS] {table_rel} found (size: {size_str}, integrity: {integrity_info})")
                else:
                    print(f"[FAIL] {table_rel} integrity check failed: {integrity_info}")
                    all_passed = False
            else:
                print(f"[PASS] {table_rel} found (size: {size_str})")
        else:
            print(f"[FAIL] {table_rel} NOT found")
            all_passed = False

    print("=" * 65)

    if all_passed:
        print("[SUCCESS] All dataset directories and clinical table files verified successfully!")
        return True
    else:
        print("[ERROR] Verification failed. Missing required dataset files or directories.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify MIMIC-IV dataset paths and structure.")
    parser.add_argument(
        "--check-integrity",
        action="store_true",
        help="Perform read integrity and header verification on all resolved clinical tables."
    )
    args = parser.parse_args()
    success = verify_dataset(check_integrity=args.check_integrity)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

