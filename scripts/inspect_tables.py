"""
Dataset table inspector and schema audit script for MIMIC-IV Ventilator System.
Audits columns, row counts, data types, missing values, and data previews across core tables.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Relative dataset path from project root
REL_DATASET_DIR = Path("mimic-iv-demo-dataset") / "mimic-iv-clinical-database-demo-2.2"

# Target tables required for Week 1 Task 1.3
TARGET_TABLES = [
    "hosp/patients.csv.gz",
    "hosp/admissions.csv.gz",
    "hosp/diagnoses_icd.csv.gz",
    "hosp/labevents.csv.gz",
    "icu/icustays.csv.gz",
    "icu/chartevents.csv.gz",
]


def resolve_table_path(base_dir: Path, rel_table_path: str) -> Path | None:
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


def inspect_table(rel_path: str, file_path: Path) -> dict:
    """
    Reads a clinical table and computes schema, row counts, missing rates, and previews.
    """
    df = pd.read_csv(file_path, low_memory=False)
    total_rows = len(df)
    total_cols = len(df.columns)

    columns_info = []
    for idx, col in enumerate(df.columns, 1):
        dtype_str = str(df[col].dtype)
        missing_count = int(df[col].isna().sum())
        missing_pct = (missing_count / total_rows * 100.0) if total_rows > 0 else 0.0
        columns_info.append({
            "index": idx,
            "column": col,
            "dtype": dtype_str,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
        })

    preview_df = df.head(2)

    return {
        "rel_path": rel_path,
        "resolved_path": file_path,
        "total_rows": total_rows,
        "total_cols": total_cols,
        "columns_info": columns_info,
        "preview_df": preview_df,
    }


def print_table_report(info: dict) -> None:
    """Prints a structured inspection report for a single clinical table."""
    print("=" * 78)
    print(f"TABLE AUDIT: {info['rel_path']}")
    print("=" * 78)
    print(f"[*] Resolved File Path: {info['resolved_path'].as_posix()}")
    print(f"[*] Total Row Count:    {info['total_rows']:,}")
    print(f"[*] Total Columns:      {info['total_cols']}")
    print("-" * 78)
    print(f"{'#':<4} {'Column Name':<28} {'Dtype':<14} {'Missing Count':<16} {'Missing %':<10}")
    print("-" * 78)
    for col in info["columns_info"]:
        print(
            f"{col['index']:<4} "
            f"{col['column']:<28} "
            f"{col['dtype']:<14} "
            f"{col['missing_count']:<16,} "
            f"{col['missing_pct']:>6.2f}%"
        )
    print("-" * 78)
    print("[*] First 2 Rows Preview:")
    print("-" * 78)
    # Display preview with readable string formatting
    preview_str = info["preview_df"].to_string(index=False)
    print(preview_str)
    print()


def run_table_inspection(base_dir: Path | None = None) -> bool:
    """Executes schema inspection across all required clinical tables."""
    project_root = Path(__file__).resolve().parent.parent
    if base_dir is None:
        base_dir = project_root / REL_DATASET_DIR

    print("=" * 78)
    print("MIMIC-IV CLINICAL DATABASE - DATASET TABLE INSPECTOR")
    print("Week 1 Task 1.3: Audit Columns, Row Counts, Dtypes & Missing Rates")
    print("=" * 78)
    print(f"[*] Dataset Directory: {base_dir.as_posix()}")

    if not base_dir.is_dir():
        print(f"[FAIL] Base dataset directory not found at: {base_dir}")
        return False

    summary_stats = []
    all_succeeded = True

    for rel_table in TARGET_TABLES:
        resolved = resolve_table_path(base_dir, rel_table)
        if not resolved:
            print(f"\n[FAIL] Table file not found: {rel_table}")
            summary_stats.append({
                "table": rel_table,
                "rows": 0,
                "cols": 0,
                "status": "FAIL (Not Found)",
            })
            all_succeeded = False
            continue

        try:
            info = inspect_table(rel_table, resolved)
            print()
            print_table_report(info)
            summary_stats.append({
                "table": rel_table,
                "rows": info["total_rows"],
                "cols": info["total_cols"],
                "status": "PASS",
            })
        except Exception as e:
            print(f"\n[FAIL] Error inspecting {rel_table}: {e}")
            summary_stats.append({
                "table": rel_table,
                "rows": 0,
                "cols": 0,
                "status": f"FAIL ({e})",
            })
            all_succeeded = False

    # Summary Table
    print("=" * 78)
    print("INSPECTION AUDIT SUMMARY")
    print("=" * 78)
    print(f"{'Status':<8} {'Table':<32} {'Rows':<12} {'Columns':<10}")
    print("-" * 78)
    for stat in summary_stats:
        status_tag = f"[{stat['status']}]"
        print(f"{status_tag:<8} {stat['table']:<32} {stat['rows']:<12,} {stat['cols']:<10}")
    print("=" * 78)

    if all_succeeded:
        print("[SUCCESS] All 6 clinical tables inspected successfully with code 0!")
        return True
    else:
        print("[FAIL] Some tables failed inspection.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Inspect columns, row counts, dtypes, and missing values in MIMIC-IV demo tables."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Optional custom path to the MIMIC-IV clinical database directory.",
    )
    args = parser.parse_args()

    success = run_table_inspection(base_dir=args.dataset_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

