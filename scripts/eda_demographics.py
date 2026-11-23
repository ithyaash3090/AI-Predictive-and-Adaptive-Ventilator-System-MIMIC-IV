"""
Exploratory Data Analysis (EDA) script for MIMIC-IV patient demographics.
Computes patient counts, gender distribution, age/anchor age statistics,
and if admissions data is available, race/ethnicity and admission type breakdowns.

Assigned for Week 1 Task 1.1 (Issue #12).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Default relative dataset path from project root
REL_DATASET_DIR = Path("mimic-iv-demo-dataset") / "mimic-iv-clinical-database-demo-2.2"
DEFAULT_OUTPUT_MD = Path("outputs") / "demographic_summary.md"


def resolve_table_path(base_dir: Path, rel_table_path: str) -> Path | None:
    """
    Resolves a table file path across standard distribution formats:
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
    parts = rel_table_path.split("/")
    if len(parts) == 2:
        folder, fname = parts
        stem = fname.replace(".csv.gz", "").replace(".csv", "")
        nested_csv = base_dir / folder / f"{stem}.csv" / f"{stem}.csv"
        if nested_csv.is_file():
            return nested_csv

    return None


def load_demographic_data(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Loads patients.csv.gz (and optionally admissions.csv.gz) from base_dir.
    Returns (patients_df, admissions_df).
    Raises FileNotFoundError if patients table cannot be resolved.
    """
    patients_path = resolve_table_path(base_dir, "hosp/patients.csv.gz")
    if not patients_path:
        raise FileNotFoundError(f"Could not resolve patients table under {base_dir}")

    patients_df = pd.read_csv(patients_path, low_memory=False)

    admissions_df = None
    admissions_path = resolve_table_path(base_dir, "hosp/admissions.csv.gz")
    if admissions_path:
        admissions_df = pd.read_csv(admissions_path, low_memory=False)

    return patients_df, admissions_df


def compute_patient_demographics(patients_df: pd.DataFrame) -> dict:
    """
    Computes patient-level summary metrics from patients.csv.
    - Unique patient count
    - Gender distribution (counts and percentages)
    - Age / anchor age statistics (mean, std, median, IQR, Q1, Q3, min, max)
    """
    if "subject_id" not in patients_df.columns:
        raise ValueError("Column 'subject_id' missing from patients dataframe")

    unique_patients = int(patients_df["subject_id"].nunique())
    total_records = len(patients_df)

    # Gender distribution
    gender_counts = patients_df["gender"].value_counts(dropna=False).to_dict()
    gender_dist = []
    for g, count in gender_counts.items():
        pct = (count / total_records * 100.0) if total_records > 0 else 0.0
        gender_dist.append({
            "gender": str(g) if pd.notna(g) else "Unknown",
            "count": int(count),
            "percentage": round(pct, 2),
        })

    # Age / anchor age statistics
    age_col = None
    if "anchor_age" in patients_df.columns:
        age_col = "anchor_age"
    elif "age" in patients_df.columns:
        age_col = "age"

    if age_col and not patients_df[age_col].dropna().empty:
        valid_ages = pd.to_numeric(patients_df[age_col], errors="coerce").dropna()
        q25 = float(valid_ages.quantile(0.25))
        q75 = float(valid_ages.quantile(0.75))
        iqr = float(q75 - q25)
        age_stats = {
            "column_used": age_col,
            "count": int(len(valid_ages)),
            "mean": round(float(valid_ages.mean()), 2),
            "std": round(float(valid_ages.std()), 2) if len(valid_ages) > 1 else 0.0,
            "median": round(float(valid_ages.median()), 2),
            "q25": round(q25, 2),
            "q75": round(q75, 2),
            "iqr": round(iqr, 2),
            "min": round(float(valid_ages.min()), 2),
            "max": round(float(valid_ages.max()), 2),
        }
    else:
        age_stats = {
            "column_used": "None",
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "iqr": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    return {
        "unique_patients": unique_patients,
        "total_records": total_records,
        "gender_distribution": gender_dist,
        "age_statistics": age_stats,
    }


def compute_admission_demographics(
    patients_df: pd.DataFrame, admissions_df: pd.DataFrame
) -> dict:
    """
    Computes demographics from merged patients and admissions tables:
    - Total admissions and distinct patients
    - Race / ethnicity frequency breakdown (counts and percentages)
    - Admission type breakdown (counts and percentages)
    """
    if "subject_id" not in admissions_df.columns:
        raise ValueError("Column 'subject_id' missing from admissions dataframe")

    merged = patients_df.merge(admissions_df, on="subject_id", how="inner")
    total_admissions = len(merged)
    unique_admitted_patients = int(merged["subject_id"].nunique())

    # Race and ethnicity breakdown
    race_dist = []
    if "race" in merged.columns:
        race_counts = merged["race"].value_counts(dropna=False)
        for race, count in race_counts.items():
            pct = (count / total_admissions * 100.0) if total_admissions > 0 else 0.0
            race_dist.append({
                "race": str(race) if pd.notna(race) else "UNKNOWN",
                "count": int(count),
                "percentage": round(pct, 2),
            })

    # Admission type breakdown
    admission_types_dist = []
    if "admission_type" in merged.columns:
        admit_counts = merged["admission_type"].value_counts(dropna=False)
        for atype, count in admit_counts.items():
            pct = (count / total_admissions * 100.0) if total_admissions > 0 else 0.0
            admission_types_dist.append({
                "admission_type": str(atype) if pd.notna(atype) else "UNKNOWN",
                "count": int(count),
                "percentage": round(pct, 2),
            })

    return {
        "total_admissions": total_admissions,
        "unique_admitted_patients": unique_admitted_patients,
        "race_distribution": race_dist,
        "admission_type_distribution": admission_types_dist,
    }


def format_terminal_output(
    patient_demo: dict,
    admission_demo: dict | None = None,
) -> str:
    """
    Formats demographic summary into a clean terminal report.
    """
    lines = []
    lines.append("=" * 78)
    lines.append("MIMIC-IV DEMOGRAPHIC SUMMARY REPORT")
    lines.append("Week 1 Task 1.1: Patient & Admission Cohort Exploratory Data Analysis")
    lines.append("=" * 78)
    lines.append(f"[*] Total Unique Patients: {patient_demo['unique_patients']:,}")
    lines.append(f"[*] Total Patient Rows:     {patient_demo['total_records']:,}")
    if admission_demo:
        lines.append(f"[*] Total Admissions:       {admission_demo['total_admissions']:,}")
        lines.append(f"[*] Admitted Patient Count: {admission_demo['unique_admitted_patients']:,}")
    lines.append("-" * 78)

    # 1. Gender distribution
    lines.append("\n[1] GENDER DISTRIBUTION")
    lines.append("-" * 78)
    lines.append(f"{'Gender':<20} {'Count':<15} {'Percentage':<15}")
    lines.append("-" * 78)
    for g in patient_demo["gender_distribution"]:
        lines.append(f"{g['gender']:<20} {g['count']:<15,} {g['percentage']:>6.2f}%")
    lines.append("-" * 78)

    # 2. Age Statistics
    age = patient_demo["age_statistics"]
    lines.append(f"\n[2] AGE / ANCHOR AGE STATISTICS (Metric Column: {age['column_used']})")
    lines.append("-" * 78)
    lines.append(f"{'Statistic':<25} {'Value':<15}")
    lines.append("-" * 78)
    lines.append(f"{'Total Valid Count':<25} {age['count']:,}")
    lines.append(f"{'Mean Age':<25} {age['mean']:.2f} years")
    lines.append(f"{'Std Deviation':<25} {age['std']:.2f} years")
    lines.append(f"{'Median Age':<25} {age['median']:.2f} years")
    lines.append(f"{'25th Percentile (Q1)':<25} {age['q25']:.2f} years")
    lines.append(f"{'75th Percentile (Q3)':<25} {age['q75']:.2f} years")
    lines.append(f"{'Interquartile Range (IQR)':<25} {age['iqr']:.2f} years")
    lines.append(f"{'Minimum Age':<25} {age['min']:.2f} years")
    lines.append(f"{'Maximum Age':<25} {age['max']:.2f} years")
    lines.append("-" * 78)

    # 3. Race & Ethnicity Breakdown (if merged)
    if admission_demo and admission_demo["race_distribution"]:
        lines.append("\n[3] RACE & ETHNICITY BREAKDOWN (Merged Admissions)")
        lines.append("-" * 78)
        lines.append(f"{'Race / Ethnicity':<45} {'Admissions':<15} {'Percentage':<15}")
        lines.append("-" * 78)
        for r in admission_demo["race_distribution"]:
            lines.append(f"{r['race']:<45} {r['count']:<15,} {r['percentage']:>6.2f}%")
        lines.append("-" * 78)

    # 4. Admission Type Breakdown (if merged)
    if admission_demo and admission_demo["admission_type_distribution"]:
        lines.append("\n[4] ADMISSION TYPE BREAKDOWN (Merged Admissions)")
        lines.append("-" * 78)
        lines.append(f"{'Admission Type':<45} {'Admissions':<15} {'Percentage':<15}")
        lines.append("-" * 78)
        for a in admission_demo["admission_type_distribution"]:
            lines.append(f"{a['admission_type']:<45} {a['count']:<15,} {a['percentage']:>6.2f}%")
        lines.append("-" * 78)

    lines.append("\n" + "=" * 78)
    lines.append("[SUCCESS] Demographic analysis complete.")
    lines.append("=" * 78)
    return "\n".join(lines)


def generate_markdown_summary(
    patient_demo: dict,
    admission_demo: dict | None = None,
    output_path: Path | None = None,
) -> str:
    """
    Renders demographic summary into a clean GitHub-flavored markdown document.
    Optionally writes to output_path if provided.
    """
    lines = []
    lines.append("# 📊 MIMIC-IV Clinical Database - Demographic Summary")
    lines.append("")
    lines.append("> **Week 1 Task 1.1 Deliverable** | Assignee: `@bhashinimanju640-star`")
    lines.append("")
    lines.append("## 1. Overview Metrics")
    lines.append("")
    lines.append("| Metric | Count | Description |")
    lines.append("| :--- | :--- | :--- |")
    lines.append(f"| **Unique Patients** | `{patient_demo['unique_patients']:,}` | Total distinct `subject_id` in `patients.csv` |")
    lines.append(f"| **Total Patient Records** | `{patient_demo['total_records']:,}` | Raw patient table record count |")
    if admission_demo:
        lines.append(f"| **Total Admissions** | `{admission_demo['total_admissions']:,}` | Total inpatient admission encounters |")
        lines.append(f"| **Unique Admitted Patients** | `{admission_demo['unique_admitted_patients']:,}` | Distinct `subject_id` linked to admissions |")
    lines.append("")

    # Gender distribution table
    lines.append("## 2. Gender Distribution")
    lines.append("")
    lines.append("| Gender | Patient Count | Percentage |")
    lines.append("| :--- | :---: | :---: |")
    for g in patient_demo["gender_distribution"]:
        lines.append(f"| **{g['gender']}** | {g['count']:,} | {g['percentage']:.2f}% |")
    lines.append("")

    # Age statistics table
    age = patient_demo["age_statistics"]
    lines.append("## 3. Patient Age & Anchor Age Statistics")
    lines.append("")
    lines.append(f"*Evaluated using column `{age['column_used']}` across `{age['count']:,}` valid patient records.*")
    lines.append("")
    lines.append("| Statistical Metric | Value | Units |")
    lines.append("| :--- | :---: | :---: |")
    lines.append(f"| **Mean Age** | `{age['mean']:.2f}` | Years |")
    lines.append(f"| **Standard Deviation** | `{age['std']:.2f}` | Years |")
    lines.append(f"| **Median Age** | `{age['median']:.2f}` | Years |")
    lines.append(f"| **25th Percentile (Q1)** | `{age['q25']:.2f}` | Years |")
    lines.append(f"| **75th Percentile (Q3)** | `{age['q75']:.2f}` | Years |")
    lines.append(f"| **Interquartile Range (IQR)** | `{age['iqr']:.2f}` | Years |")
    lines.append(f"| **Minimum Age** | `{age['min']:.2f}` | Years |")
    lines.append(f"| **Maximum Age** | `{age['max']:.2f}` | Years |")
    lines.append("")

    # Race & Ethnicity distribution table
    if admission_demo and admission_demo["race_distribution"]:
        lines.append("## 4. Race & Ethnicity Breakdown (Hospital Admissions)")
        lines.append("")
        lines.append("| Race / Ethnicity Category | Admission Encounters | Percentage |")
        lines.append("| :--- | :---: | :---: |")
        for r in admission_demo["race_distribution"]:
            lines.append(f"| {r['race']} | {r['count']:,} | {r['percentage']:.2f}% |")
        lines.append("")

    # Admission Type breakdown table
    if admission_demo and admission_demo["admission_type_distribution"]:
        lines.append("## 5. Admission Type Breakdown")
        lines.append("")
        lines.append("| Admission Type | Encounter Count | Percentage |")
        lines.append("| :--- | :---: | :---: |")
        for a in admission_demo["admission_type_distribution"]:
            lines.append(f"| {a['admission_type']} | {a['count']:,} | {a['percentage']:.2f}% |")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by `scripts/eda_demographics.py` for MIMIC-IV Ventilator System.*")
    lines.append("")

    content = "\n".join(lines)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

    return content


def run_demographics_eda(
    dataset_dir: Path | None = None,
    output_md: Path | None = None,
    save_markdown: bool = True,
    quiet: bool = False,
) -> bool:
    """
    Main execution pipeline for demographic EDA.
    Loads dataset, computes metrics, prints to terminal, and writes markdown summary.
    """
    project_root = Path(__file__).resolve().parent.parent
    base_dir = dataset_dir if dataset_dir else (project_root / REL_DATASET_DIR)

    if not base_dir.is_dir():
        print(f"[ERROR] Dataset directory not found at: {base_dir}")
        return False

    try:
        patients_df, admissions_df = load_demographic_data(base_dir)
    except Exception as e:
        print(f"[ERROR] Failed to load demographic data: {e}")
        return False

    patient_demo = compute_patient_demographics(patients_df)
    admission_demo = (
        compute_admission_demographics(patients_df, admissions_df)
        if admissions_df is not None
        else None
    )

    if not quiet:
        terminal_report = format_terminal_output(patient_demo, admission_demo)
        print(terminal_report)

    if save_markdown:
        out_file = output_md if output_md else (project_root / DEFAULT_OUTPUT_MD)
        generate_markdown_summary(patient_demo, admission_demo, output_path=out_file)
        if not quiet:
            print(f"\n[INFO] Demographic summary successfully written to: {out_file.as_posix()}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compute exploratory demographic summaries for MIMIC-IV demo dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Path to MIMIC-IV clinical database directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output markdown summary file (default: outputs/demographic_summary.md).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write markdown summary file to disk.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress terminal summary output.",
    )
    args = parser.parse_args()

    success = run_demographics_eda(
        dataset_dir=args.dataset_dir,
        output_md=args.output,
        save_markdown=not args.no_save,
        quiet=args.quiet,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
