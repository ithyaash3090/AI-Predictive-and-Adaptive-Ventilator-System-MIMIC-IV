"""
Exploratory Comorbidity Analysis Script for MIMIC-IV Ventilator System.
Maps ICD-9 and ICD-10 diagnoses to clinical risk flags (Acute Respiratory Failure,
ARDS, Pneumonia, COPD, Sepsis), computes prevalence rates, diagnosis count distributions
per patient stay, and pairwise comorbidity co-occurrence matrices.

Outputs formatted summary tables in the terminal and optionally writes a markdown
report to outputs/comorbidity_summary.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Default relative paths
REL_DATASET_DIR = Path("mimic-iv-demo-dataset") / "mimic-iv-clinical-database-demo-2.2"
DEFAULT_OUTPUT_MD = Path("outputs") / "comorbidity_summary.md"

# Condition definitions with display names and mapping patterns
CONDITION_DEFINITIONS = [
    {
        "key": "acute_respiratory_failure",
        "name": "Acute Respiratory Failure",
        "title_regex": r"\brespiratory failure\b",
        "code_prefixes": ("51881", "51884", "51851", "J960", "J962", "J969"),
    },
    {
        "key": "ards",
        "name": "ARDS / Acute Respiratory Distress",
        "title_regex": r"\bards\b|\bacute respiratory distress\b",
        "code_prefixes": ("51882", "J80", "R0603"),
    },
    {
        "key": "pneumonia",
        "name": "Pneumonia",
        "title_regex": r"\bpneumonia\b",
        "code_prefixes": (
            "480", "481", "482", "483", "484", "485", "486", "4870", "48811",
            "J12", "J13", "J14", "J15", "J16", "J17", "J18", "J95851", "99731",
        ),
    },
    {
        "key": "copd",
        "name": "COPD / Chronic Obstructive Pulmonary",
        "title_regex": r"\bchronic obstructive pulmonary\b|\bcopd\b|\bemphysema\b",
        "code_prefixes": ("491", "492", "496", "J41", "J42", "J43", "J44"),
    },
    {
        "key": "sepsis",
        "name": "Sepsis / Septic Shock",
        "title_regex": r"\bsepsis\b|\bseptic shock\b|\bsevere sepsis\b",
        "code_prefixes": ("038", "99591", "99592", "78552", "A40", "A41", "R652"),
    },
]


def resolve_table_path(base_dir: Path, rel_table_path: str) -> Optional[Path]:
    """
    Resolves table file path across standard distribution formats:
    - Direct compressed: base_dir / rel_table_path (e.g. hosp/diagnoses_icd.csv.gz)
    - Uncompressed flat: base_dir / rel_table_path.csv (e.g. hosp/diagnoses_icd.csv)
    - Git demo nested: base_dir / folder / stem.csv / stem.csv
    """
    direct = base_dir / rel_table_path
    if direct.is_file():
        return direct

    flat_csv = base_dir / rel_table_path.replace(".csv.gz", ".csv")
    if flat_csv.is_file():
        return flat_csv

    parts = rel_table_path.split("/")
    if len(parts) == 2:
        folder, fname = parts
        stem = fname.replace(".csv.gz", "").replace(".csv", "")
        nested_csv = base_dir / folder / f"{stem}.csv" / f"{stem}.csv"
        if nested_csv.is_file():
            return nested_csv

    return None


def load_diagnosis_data(
    base_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads diagnoses_icd and d_icd_diagnoses from base_dir.
    Returns (df_diagnoses, df_d_diagnoses).
    Raises FileNotFoundError if required tables cannot be found.
    """
    diag_path = resolve_table_path(base_dir, "hosp/diagnoses_icd.csv.gz")
    d_diag_path = resolve_table_path(base_dir, "hosp/d_icd_diagnoses.csv.gz")

    if not diag_path or not diag_path.is_file():
        raise FileNotFoundError(f"Could not resolve diagnoses_icd table under {base_dir}")
    if not d_diag_path or not d_diag_path.is_file():
        raise FileNotFoundError(f"Could not resolve d_icd_diagnoses table under {base_dir}")

    df_diag = pd.read_csv(diag_path, low_memory=False)
    df_d_diag = pd.read_csv(d_diag_path, low_memory=False)
    return df_diag, df_d_diag


def map_icd_conditions(
    df_diagnoses: pd.DataFrame,
    df_d_diagnoses: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges diagnostic records with ICD definitions and assigns boolean flags
    for each clinical risk condition.
    Handles empty inputs, missing descriptions, and cleans code formats.
    """
    if df_diagnoses.empty:
        cols = ["subject_id", "hadm_id", "icd_code", "icd_version", "long_title"] + [
            cfg["key"] for cfg in CONDITION_DEFINITIONS
        ]
        return pd.DataFrame(columns=cols)

    # Ensure required columns exist
    for col in ["subject_id", "hadm_id", "icd_code", "icd_version"]:
        if col not in df_diagnoses.columns:
            raise ValueError(f"Required column '{col}' missing from diagnoses dataframe")

    # Merge with dictionary table
    if not df_d_diagnoses.empty and "long_title" in df_d_diagnoses.columns:
        merged = df_diagnoses.merge(
            df_d_diagnoses[["icd_code", "icd_version", "long_title"]],
            on=["icd_code", "icd_version"],
            how="left",
        )
    else:
        merged = df_diagnoses.copy()
        merged["long_title"] = ""

    merged["long_title"] = merged["long_title"].fillna("").astype(str)
    title_lower = merged["long_title"].str.lower()
    clean_code = (
        merged["icd_code"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.upper()
    )

    for cfg in CONDITION_DEFINITIONS:
        k = cfg["key"]
        text_match = title_lower.str.contains(cfg["title_regex"], regex=True)
        code_match = clean_code.str.startswith(cfg["code_prefixes"])
        merged[k] = (text_match | code_match).astype(int)

    return merged


def compute_comorbidity_metrics(
    mapped_df: pd.DataFrame,
) -> Dict:
    """
    Aggregates condition flags and diagnosis counts at the patient stay (hadm_id) level:
    - Total stays and total patients
    - Prevalence (counts and percentages) for each condition
    - Diagnosis count distribution statistics per stay
    - Co-occurrence matrix across conditions
    """
    if mapped_df.empty or "hadm_id" not in mapped_df.columns:
        empty_matrix = pd.DataFrame(
            0,
            index=[c["key"] for c in CONDITION_DEFINITIONS],
            columns=[c["key"] for c in CONDITION_DEFINITIONS],
        )
        return {
            "total_records": 0,
            "total_stays": 0,
            "total_patients": 0,
            "prevalence": [
                {
                    "key": c["key"],
                    "name": c["name"],
                    "stay_count": 0,
                    "stay_pct": 0.0,
                    "patient_count": 0,
                    "patient_pct": 0.0,
                }
                for c in CONDITION_DEFINITIONS
            ],
            "stay_diagnosis_stats": {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "q25": 0.0,
                "q75": 0.0,
                "iqr": 0.0,
                "min": 0.0,
                "max": 0.0,
            },
            "cooccurrence_counts": empty_matrix,
            "cooccurrence_rates": empty_matrix.astype(float),
        }

    total_records = len(mapped_df)
    total_stays = int(mapped_df["hadm_id"].dropna().nunique())
    total_patients = int(mapped_df["subject_id"].dropna().nunique())

    # Aggregate by hadm_id (patient stay)
    cond_keys = [c["key"] for c in CONDITION_DEFINITIONS]
    
    agg_funcs = {k: "max" for k in cond_keys}
    agg_funcs["subject_id"] = "first"
    agg_funcs["icd_code"] = "count"  # Total diagnosis count for this stay

    stay_summary = (
        mapped_df.groupby("hadm_id", as_index=False)
        .agg(agg_funcs)
        .rename(columns={"icd_code": "total_diagnoses"})
    )

    # 1. Prevalence calculations
    prevalence_list = []
    for cfg in CONDITION_DEFINITIONS:
        k = cfg["key"]
        stay_cnt = int(stay_summary[k].sum())
        stay_pct = (stay_cnt / total_stays * 100.0) if total_stays > 0 else 0.0

        # Unique patients with condition
        cond_pts = stay_summary[stay_summary[k] == 1]["subject_id"].nunique()
        patient_cnt = int(cond_pts)
        patient_pct = (patient_cnt / total_patients * 100.0) if total_patients > 0 else 0.0

        prevalence_list.append({
            "key": k,
            "name": cfg["name"],
            "stay_count": stay_cnt,
            "stay_pct": round(stay_pct, 2),
            "patient_count": patient_cnt,
            "patient_pct": round(patient_pct, 2),
        })

    # 2. Total diagnoses per patient stay distribution statistics
    diag_counts = stay_summary["total_diagnoses"].astype(float)
    if not diag_counts.empty:
        q25 = float(diag_counts.quantile(0.25))
        q75 = float(diag_counts.quantile(0.75))
        stay_diag_stats = {
            "count": int(len(diag_counts)),
            "mean": round(float(diag_counts.mean()), 2),
            "std": round(float(diag_counts.std()), 2) if len(diag_counts) > 1 else 0.0,
            "median": round(float(diag_counts.median()), 2),
            "q25": round(q25, 2),
            "q75": round(q75, 2),
            "iqr": round(float(q75 - q25), 2),
            "min": round(float(diag_counts.min()), 2),
            "max": round(float(diag_counts.max()), 2),
        }
    else:
        stay_diag_stats = {
            "count": 0, "mean": 0.0, "std": 0.0, "median": 0.0,
            "q25": 0.0, "q75": 0.0, "iqr": 0.0, "min": 0.0, "max": 0.0,
        }

    # 3. Co-occurrence matrix
    cooc_counts = pd.DataFrame(index=cond_keys, columns=cond_keys, dtype=int)
    cooc_rates = pd.DataFrame(index=cond_keys, columns=cond_keys, dtype=float)

    for r in cond_keys:
        for c in cond_keys:
            joint = int(((stay_summary[r] == 1) & (stay_summary[c] == 1)).sum())
            cooc_counts.loc[r, c] = joint
            # Rate relative to total patient stays
            pct = (joint / total_stays * 100.0) if total_stays > 0 else 0.0
            cooc_rates.loc[r, c] = round(pct, 2)

    return {
        "total_records": total_records,
        "total_stays": total_stays,
        "total_patients": total_patients,
        "prevalence": prevalence_list,
        "stay_diagnosis_stats": stay_diag_stats,
        "cooccurrence_counts": cooc_counts,
        "cooccurrence_rates": cooc_rates,
    }


def format_terminal_output(metrics: Dict) -> str:
    """
    Formats the comorbidity summary into clean, readable terminal tables.
    """
    lines = []
    lines.append("=" * 86)
    lines.append("MIMIC-IV EXPLORATORY COMORBIDITY ANALYSIS REPORT")
    lines.append("Clinical Risk Flags: Acute Respiratory Failure, ARDS, Pneumonia, COPD, Sepsis")
    lines.append("=" * 86)
    lines.append(f"[*] Total Diagnosis Records:  {metrics['total_records']:,}")
    lines.append(f"[*] Total Hospital Stays:    {metrics['total_stays']:,}")
    lines.append(f"[*] Total Unique Patients:   {metrics['total_patients']:,}")
    lines.append("-" * 86)

    # 1. Prevalence Table
    lines.append("\n[1] CLINICAL RISK FLAG PREVALENCE")
    lines.append("-" * 86)
    lines.append(
        f"{'Condition':<40} "
        f"{'Stay Count':<14} "
        f"{'Stay %':<12} "
        f"{'Patients':<12} "
        f"{'Patient %':<10}"
    )
    lines.append("-" * 86)
    for p in metrics["prevalence"]:
        lines.append(
            f"{p['name']:<40} "
            f"{p['stay_count']:<14,} "
            f"{p['stay_pct']:>6.2f}%     "
            f"{p['patient_count']:<12,} "
            f"{p['patient_pct']:>6.2f}%"
        )
    lines.append("-" * 86)

    # 2. Diagnosis Count Distribution
    stats = metrics["stay_diagnosis_stats"]
    lines.append("\n[2] DIAGNOSES PER PATIENT STAY DISTRIBUTION")
    lines.append("-" * 86)
    lines.append(f"{'Statistic':<35} {'Value':<15}")
    lines.append("-" * 86)
    lines.append(f"{'Total Evaluated Stays':<35} {stats['count']:,}")
    lines.append(f"{'Mean Diagnoses per Stay':<35} {stats['mean']:.2f}")
    lines.append(f"{'Standard Deviation':<35} {stats['std']:.2f}")
    lines.append(f"{'Median Diagnoses':<35} {stats['median']:.2f}")
    lines.append(f"{'25th Percentile (Q1)':<35} {stats['q25']:.2f}")
    lines.append(f"{'75th Percentile (Q3)':<35} {stats['q75']:.2f}")
    lines.append(f"{'Interquartile Range (IQR)':<35} {stats['iqr']:.2f}")
    lines.append(f"{'Minimum Diagnoses':<35} {stats['min']:.2f}")
    lines.append(f"{'Maximum Diagnoses':<35} {stats['max']:.2f}")
    lines.append("-" * 86)

    # 3. Pairwise Co-Occurrence Matrix (Counts and Percentages)
    short_names = {
        "acute_respiratory_failure": "Resp Fail",
        "ards": "ARDS",
        "pneumonia": "Pneumonia",
        "copd": "COPD",
        "sepsis": "Sepsis",
    }
    lines.append("\n[3] PAIRWISE COMORBIDITY CO-OCCURRENCE (Stays Count / Stay %)")
    lines.append("-" * 86)
    header = f"{'Condition':<18}" + "".join(f"{short_names[c]:>13}" for c in short_names)
    lines.append(header)
    lines.append("-" * 86)

    counts_df = metrics["cooccurrence_counts"]
    rates_df = metrics["cooccurrence_rates"]
    for r in short_names:
        row_str = f"{short_names[r]:<18}"
        for c in short_names:
            cnt = int(counts_df.loc[r, c])
            pct = float(rates_df.loc[r, c])
            row_str += f"{cnt:>5} ({pct:>4.1f}%)"
        lines.append(row_str)
    lines.append("-" * 86)

    lines.append("\n" + "=" * 86)
    lines.append("[SUCCESS] Exploratory comorbidity analysis complete.")
    lines.append("=" * 86)
    return "\n".join(lines)


def generate_markdown_summary(
    metrics: Dict,
    output_path: Optional[Path] = None,
) -> str:
    """
    Renders the comorbidity analysis into a clean GitHub-flavored markdown report.
    Optionally writes to output_path.
    """
    lines = []
    lines.append("# 📊 MIMIC-IV Clinical Database - Comorbidity Analysis Summary")
    lines.append("")
    lines.append("> **Exploratory Comorbidity Analysis Deliverable** | MIMIC-IV Ventilator System")
    lines.append("")
    lines.append("## 1. Cohort Diagnostic Overview")
    lines.append("")
    lines.append("| Metric | Count | Description |")
    lines.append("| :--- | :--- | :--- |")
    lines.append(f"| **Total Diagnosis Records** | `{metrics['total_records']:,}` | Raw ICD diagnosis entries |")
    lines.append(f"| **Hospital Patient Stays** | `{metrics['total_stays']:,}` | Distinct hospital admission encounters (`hadm_id`) |")
    lines.append(f"| **Unique Patients** | `{metrics['total_patients']:,}` | Distinct patients (`subject_id`) |")
    lines.append("")

    # Prevalence table
    lines.append("## 2. Clinical Risk Flag Prevalence")
    lines.append("")
    lines.append("| Clinical Risk Flag | Stay Count (`hadm_id`) | Stay % | Patient Count (`subject_id`) | Patient % |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for p in metrics["prevalence"]:
        lines.append(
            f"| **{p['name']}** | {p['stay_count']:,} | {p['stay_pct']:.2f}% | "
            f"{p['patient_count']:,} | {p['patient_pct']:.2f}% |"
        )
    lines.append("")

    # Diagnosis count statistics
    stats = metrics["stay_diagnosis_stats"]
    lines.append("## 3. Total Diagnoses per Patient Stay Distribution")
    lines.append("")
    lines.append("| Statistical Metric | Value | Units |")
    lines.append("| :--- | :---: | :---: |")
    lines.append(f"| **Mean Diagnoses** | `{stats['mean']:.2f}` | Diagnoses / Stay |")
    lines.append(f"| **Standard Deviation** | `{stats['std']:.2f}` | Diagnoses / Stay |")
    lines.append(f"| **Median Diagnoses** | `{stats['median']:.2f}` | Diagnoses / Stay |")
    lines.append(f"| **25th Percentile (Q1)** | `{stats['q25']:.2f}` | Diagnoses / Stay |")
    lines.append(f"| **75th Percentile (Q3)** | `{stats['q75']:.2f}` | Diagnoses / Stay |")
    lines.append(f"| **Interquartile Range (IQR)** | `{stats['iqr']:.2f}` | Diagnoses / Stay |")
    lines.append(f"| **Minimum Diagnoses** | `{stats['min']:.2f}` | Diagnoses / Stay |")
    lines.append(f"| **Maximum Diagnoses** | `{stats['max']:.2f}` | Diagnoses / Stay |")
    lines.append("")

    # Co-occurrence matrix
    short_names = {
        "acute_respiratory_failure": "Resp Fail",
        "ards": "ARDS",
        "pneumonia": "Pneumonia",
        "copd": "COPD",
        "sepsis": "Sepsis",
    }
    counts_df = metrics["cooccurrence_counts"]
    rates_df = metrics["cooccurrence_rates"]

    lines.append("## 4. Pairwise Comorbidity Co-Occurrence Matrix")
    lines.append("")
    lines.append("*Values represented as: `Joint Stays Count (Percentage of Total Stays)`*")
    lines.append("")
    cols_header = "| Condition | " + " | ".join(short_names.values()) + " |"
    sep_header = "| :--- | " + " | ".join([":---:"] * len(short_names)) + " |"
    lines.append(cols_header)
    lines.append(sep_header)

    for r in short_names:
        row_str = f"| **{short_names[r]}** | "
        cells = []
        for c in short_names:
            cnt = int(counts_df.loc[r, c])
            pct = float(rates_df.loc[r, c])
            cells.append(f"{cnt:,} ({pct:.1f}%)")
        row_str += " | ".join(cells) + " |"
        lines.append(row_str)
    lines.append("")

    lines.append("---")
    lines.append("*Generated by `scripts/eda_comorbidities.py` for MIMIC-IV Ventilator System.*")
    lines.append("")

    content = "\n".join(lines)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

    return content


def run_comorbidity_eda(
    dataset_dir: Optional[Path] = None,
    output_md: Optional[Path] = None,
    save_markdown: bool = True,
    quiet: bool = False,
) -> bool:
    """
    Main orchestrator function for exploratory comorbidity analysis.
    Loads diagnoses data, maps risk flags, computes metrics, prints summary,
    and optionally writes markdown report.
    """
    project_root = Path(__file__).resolve().parent.parent
    base_dir = dataset_dir if dataset_dir else (project_root / REL_DATASET_DIR)

    if not base_dir.is_dir():
        print(f"[ERROR] Dataset directory not found at: {base_dir}")
        return False

    try:
        df_diag, df_d_diag = load_diagnosis_data(base_dir)
    except Exception as e:
        print(f"[ERROR] Failed to load diagnoses tables: {e}")
        return False

    mapped_df = map_icd_conditions(df_diag, df_d_diag)
    metrics = compute_comorbidity_metrics(mapped_df)

    if not quiet:
        text_report = format_terminal_output(metrics)
        print(text_report)

    if save_markdown:
        out_file = output_md if output_md else (project_root / DEFAULT_OUTPUT_MD)
        generate_markdown_summary(metrics, output_path=out_file)
        if not quiet:
            print(f"\n[INFO] Comorbidity summary report saved successfully to: {out_file.as_posix()}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Perform exploratory comorbidity analysis and ICD condition mapping for MIMIC-IV."
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
        help="Path for output markdown summary file (default: outputs/comorbidity_summary.md).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save markdown summary report to disk.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress terminal summary output.",
    )
    args = parser.parse_args()

    success = run_comorbidity_eda(
        dataset_dir=args.dataset_dir,
        output_md=args.output,
        save_markdown=not args.no_save,
        quiet=args.quiet,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
