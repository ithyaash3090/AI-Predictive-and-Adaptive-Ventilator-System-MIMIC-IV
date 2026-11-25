"""
Patient-Level Cross-Validation Splitter Blueprint (Week 1 Task 1.12).
Evaluates GroupKFold patient-level partitioning across the MIMIC-IV cohort,
verifies zero patient leakage, and audits fold balance metrics.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.trainer import PatientGroupKFoldSplitter


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments for cross-validation splitter evaluation."""
    parser = argparse.ArgumentParser(
        description="Patient-level GroupKFold splitter evaluation and leakage verification utility."
    )
    parser.add_argument(
        "--cohort-path",
        type=str,
        default="outputs/processed_mimic_cohort.csv",
        help="Path to processed MIMIC-IV cohort CSV.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of GroupKFold cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default="subject_id",
        help="Grouping column guaranteeing patient-level isolation (default: subject_id).",
    )
    parser.add_argument(
        "--stay-col",
        type=str,
        default="stay_id",
        help="ICU stay identifier column (default: stay_id).",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="target_deterioration_lead",
        help="Binary target column for deterioration prediction (default: target_deterioration_lead).",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="outputs/splitter_audit.md",
        help="Path to save markdown audit report (default: outputs/splitter_audit.md).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Raise exception and exit with code 1 if any data leakage is detected.",
    )
    return parser.parse_args(argv)


def load_cohort(cohort_path: str, target_col: str) -> pd.DataFrame:
    """Loads and validates the cohort dataset."""
    if not os.path.exists(cohort_path):
        raise FileNotFoundError(f"Cohort dataset not found at: {cohort_path}")

    df = pd.read_csv(cohort_path)

    # Check target column with fallback to deterioration_next_4h
    if target_col not in df.columns:
        fallback = "deterioration_next_4h"
        if fallback in df.columns:
            target_col = fallback
        else:
            raise KeyError(f"Neither '{target_col}' nor '{fallback}' found in cohort columns.")

    required_cols = ["subject_id", "stay_id", target_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in cohort: {missing}")

    return df


def evaluate_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = "subject_id",
    stay_col: str = "stay_id",
    target_col: str = "target_deterioration_lead",
) -> List[Dict[str, Any]]:
    """
    Executes patient-level GroupKFold splitting and computes fold balance metrics.
    Verifies that no subject appears in both training and validation sets.
    """
    # Resolve target column
    if target_col not in df.columns and "deterioration_next_4h" in df.columns:
        target_col = "deterioration_next_4h"

    splitter = PatientGroupKFoldSplitter(n_splits=n_splits)
    splits = splitter.split(df, group_col=group_col, n_splits=n_splits)

    total_patients = set(df[group_col].unique())
    all_val_subjects: List[Any] = []
    fold_results: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_subjects = set(train_df[group_col].unique())
        val_subjects = set(val_df[group_col].unique())

        # Assert zero leakage
        overlap = train_subjects.intersection(val_subjects)
        if overlap:
            raise ValueError(
                f"Data leakage in Fold {fold_idx}: {len(overlap)} subjects present in both train and val: {overlap}"
            )

        all_val_subjects.extend(list(val_subjects))

        # Stay counts
        train_stays = train_df[stay_col].nunique()
        val_stays = val_df[stay_col].nunique()

        # Row counts
        train_rows = len(train_df)
        val_rows = len(val_df)

        # Target class statistics
        val_pos = int(val_df[target_col].sum())
        val_neg = val_rows - val_pos
        val_pos_rate = float(val_pos / val_rows) if val_rows > 0 else 0.0

        train_pos = int(train_df[target_col].sum())
        train_neg = train_rows - train_pos
        train_pos_rate = float(train_pos / train_rows) if train_rows > 0 else 0.0

        fold_results.append({
            "fold": fold_idx,
            "train_patients": len(train_subjects),
            "val_patients": len(val_subjects),
            "overlap_patients": len(overlap),
            "train_stays": train_stays,
            "val_stays": val_stays,
            "train_rows": train_rows,
            "val_rows": val_rows,
            "train_pos_rate": train_pos_rate,
            "val_pos": val_pos,
            "val_neg": val_neg,
            "val_pos_rate": val_pos_rate,
            "leakage_status": "PASS" if len(overlap) == 0 else "FAIL",
        })

    # Verify that validation subjects form a complete partition of all cohort patients
    assert len(all_val_subjects) == len(total_patients), "Validation subjects count mismatch"
    assert set(all_val_subjects) == total_patients, "Some cohort subjects were omitted from validation"

    return fold_results


def format_terminal_report(
    results: List[Dict[str, Any]],
    n_splits: int,
    total_patients: int,
    total_stays: int,
    total_rows: int,
) -> str:
    """Formats a clean terminal summary table."""
    lines = [
        "=" * 82,
        "MIMIC-IV PATIENT-LEVEL CROSS-VALIDATION SPLITTER BLUEPRINT (Task 1.12)",
        "=" * 82,
        f"[*] Cohort Summary : {total_patients} Patients | {total_stays} ICU Stays | {total_rows:,} Hourly Rows",
        f"[*] Partitioning   : GroupKFold (k={n_splits}) strictly grouped by subject_id",
        f"[*] Leakage Policy : Strict zero-leakage enforcement across train/validation folds",
        "-" * 82,
        f"{'Fold':<6} {'Train Pts':<11} {'Val Pts':<9} {'Val Stays':<11} {'Val Rows':<10} {'Val Pos Rate':<14} {'Leakage Status':<15}",
        "-" * 82,
    ]

    val_pos_rates = []
    val_row_counts = []

    for r in results:
        val_pos_rates.append(r["val_pos_rate"])
        val_row_counts.append(r["val_rows"])
        lines.append(
            f"[{r['fold']}/{n_splits}] "
            f"{r['train_patients']:<11} "
            f"{r['val_patients']:<9} "
            f"{r['val_stays']:<11} "
            f"{r['val_rows']:<10} "
            f"{r['val_pos_rate']:>6.2%}        "
            f"[{r['leakage_status']}] (Overlap: {r['overlap_patients']})"
        )

    lines.extend([
        "-" * 82,
        "[*] Fold Balance Metrics:",
        f"    - Mean Validation Positive Rate : {np.mean(val_pos_rates):.2%} (+/- {np.std(val_pos_rates):.2%})",
        f"    - Mean Validation Row Count     : {np.mean(val_row_counts):.1f} (+/- {np.std(val_row_counts):.1f})",
        f"    - Total Unique Patients Tested  : {total_patients} (100% evaluated without replacement)",
        f"    - Overlapping Subjects Detected : 0 (Complete patient isolation confirmed)",
        "=" * 82,
        "[SUCCESS] All cross-validation folds verified with zero patient leakage!",
        "=" * 82,
    ])
    return "\n".join(lines)


def export_markdown_report(
    results: List[Dict[str, Any]],
    output_path: str,
    total_patients: int,
    total_stays: int,
    total_rows: int,
    n_splits: int,
) -> None:
    """Exports structured markdown summary table."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    val_pos_rates = [r["val_pos_rate"] for r in results]

    md_lines = [
        "# Patient-Level Cross-Validation Splitter Audit Report",
        "",
        "## 1. Overview",
        f"- **Cohort Dataset**: {total_patients} unique patients, {total_stays} ICU stays, {total_rows:,} hourly observations",
        f"- **Cross-Validation Scheme**: GroupKFold (k={n_splits}) grouped strictly on `subject_id`",
        "- **Evaluation Objective**: Prevent data leakage across multiple admissions/stays of the same patient",
        "- **Leakage Outcome**: Confirmed zero overlapping subjects across all folds",
        "",
        "## 2. Fold Partitioning Table",
        "",
        "| Fold | Train Patients | Val Patients | Val Stays | Val Rows | Val Positive Count | Val Positive Rate | Leakage Status |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        md_lines.append(
            f"| Fold {r['fold']} | {r['train_patients']} | {r['val_patients']} | {r['val_stays']} | "
            f"{r['val_rows']} | {r['val_pos']} | {r['val_pos_rate']:.2%} | {r['leakage_status']} |"
        )

    md_lines.extend([
        "",
        "## 3. Fold Balance Summary",
        f"- **Average Positive Rate**: {np.mean(val_pos_rates):.2%} (+/- {np.std(val_pos_rates):.2%})",
        f"- **Total Patients Evaluated**: {total_patients} unique individuals",
        "- **Patient Isolation**: Verified strict disjoint sets ($S_{train} \\cap S_{val} = \\emptyset$ for all folds)",
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


def main(argv: Optional[List[str]] = None) -> int:
    """Main execution function."""
    args = parse_args(argv)
    df = load_cohort(args.cohort_path, args.target_col)

    total_patients = df[args.group_col].nunique()
    total_stays = df[args.stay_col].nunique()
    total_rows = len(df)

    results = evaluate_splits(
        df=df,
        n_splits=args.n_splits,
        group_col=args.group_col,
        stay_col=args.stay_col,
        target_col=args.target_col,
    )

    report_text = format_terminal_report(
        results=results,
        n_splits=args.n_splits,
        total_patients=total_patients,
        total_stays=total_stays,
        total_rows=total_rows,
    )
    print(report_text)

    if args.output_report:
        export_markdown_report(
            results=results,
            output_path=args.output_report,
            total_patients=total_patients,
            total_stays=total_stays,
            total_rows=total_rows,
            n_splits=args.n_splits,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
