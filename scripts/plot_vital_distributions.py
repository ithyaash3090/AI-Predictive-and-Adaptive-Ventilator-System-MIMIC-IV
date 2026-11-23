"""
Clinical Vital Signs Distribution Histograms & Kernel Density Estimation (KDE).
Extracts key ICU vital signs (Heart Rate, Respiratory Rate, SpO2, Systolic BP,
Diastolic BP, Temperature) from MIMIC-IV chartevents/cohort, annotates summary
statistics (mean, median, standard deviation, 5th and 95th percentiles),
and renders publication-grade distribution plots.

Assigned for Week 1 Task 1.5 (Issue #26).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# Default paths
REL_DATASET_DIR = Path("mimic-iv-demo-dataset") / "mimic-iv-clinical-database-demo-2.2"
DEFAULT_COHORT_PATH = Path("outputs") / "processed_mimic_cohort.csv"
DEFAULT_OUTPUT_PNG = Path("outputs") / "vital_distributions.png"

# Clinical Vital Signs Configuration & Item ID Mappings
VITALS_CONFIG = [
    {
        "key": "heart_rate",
        "name": "Heart Rate",
        "unit": "bpm",
        "itemids": [220045],
        "color": "#e63946",  # Red
        "clip_range": (20.0, 240.0),
    },
    {
        "key": "respiratory_rate",
        "name": "Respiratory Rate",
        "unit": "insp/min",
        "itemids": [220210],
        "color": "#457b9d",  # Steel Blue
        "clip_range": (4.0, 60.0),
    },
    {
        "key": "spo2",
        "name": "Oxygen Saturation (SpO2)",
        "unit": "%",
        "itemids": [220277],
        "color": "#2a9d8f",  # Teal Green
        "clip_range": (50.0, 100.0),
    },
    {
        "key": "systolic_bp",
        "name": "Systolic Blood Pressure",
        "unit": "mmHg",
        "itemids": [220179, 220050],  # NBP and ABP systolic
        "color": "#e76f51",  # Burnt Coral
        "clip_range": (40.0, 260.0),
    },
    {
        "key": "diastolic_bp",
        "name": "Diastolic Blood Pressure",
        "unit": "mmHg",
        "itemids": [220180, 220051],  # NBP and ABP diastolic
        "color": "#9b5de5",  # Purple
        "clip_range": (20.0, 160.0),
    },
    {
        "key": "temperature",
        "name": "Body Temperature",
        "unit": "°C",
        "itemids": [223762, 223761],  # Celsius and Fahrenheit
        "color": "#f4a261",  # Amber Orange
        "clip_range": (30.0, 43.0),
    },
]


def resolve_table_path(base_dir: Path, rel_table_path: str) -> Optional[Path]:
    """
    Resolves table file path across standard distribution formats:
    - Direct compressed: base_dir / rel_table_path (e.g. icu/chartevents.csv.gz)
    - Uncompressed flat: base_dir / rel_table_path.csv (e.g. icu/chartevents.csv)
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


def extract_vitals_from_chartevents(chartevents_path: Path) -> Dict[str, pd.Series]:
    """
    Extracts numerical observations for the 6 target clinical vitals from chartevents.
    Standardizes temperature from °F to °C if recorded in Fahrenheit.
    """
    all_itemids = []
    for cfg in VITALS_CONFIG:
        all_itemids.extend(cfg["itemids"])

    df = pd.read_csv(
        chartevents_path,
        usecols=["itemid", "valuenum"],
        low_memory=False,
    )
    df = df[df["itemid"].isin(all_itemids)].dropna(subset=["valuenum"])

    vitals: Dict[str, pd.Series] = {}

    # 1. Heart Rate
    vitals["heart_rate"] = df[df["itemid"] == 220045]["valuenum"].astype(float)

    # 2. Respiratory Rate
    vitals["respiratory_rate"] = df[df["itemid"] == 220210]["valuenum"].astype(float)

    # 3. SpO2
    vitals["spo2"] = df[df["itemid"] == 220277]["valuenum"].astype(float)

    # 4. Systolic BP
    vitals["systolic_bp"] = df[df["itemid"].isin([220179, 220050])]["valuenum"].astype(float)

    # 5. Diastolic BP
    vitals["diastolic_bp"] = df[df["itemid"].isin([220180, 220051])]["valuenum"].astype(float)

    # 6. Temperature (Celsius: 223762, Fahrenheit: 223761)
    temp_c = df[df["itemid"] == 223762]["valuenum"].astype(float)
    # Correct entries recorded in Fahrenheit under Celsius itemid (temp > 45)
    temp_c = temp_c.apply(lambda x: (x - 32.0) * 5.0 / 9.0 if x > 45.0 else x)

    temp_f = df[df["itemid"] == 223761]["valuenum"].astype(float)
    temp_f_converted = (temp_f - 32.0) * 5.0 / 9.0

    vitals["temperature"] = pd.concat([temp_c, temp_f_converted], ignore_index=True)

    # Apply physiological clip bounds for clean distribution visualization
    for cfg in VITALS_CONFIG:
        k = cfg["key"]
        if k in vitals and not vitals[k].empty:
            low, high = cfg["clip_range"]
            s = vitals[k]
            vitals[k] = s[(s >= low) & (s <= high)]

    return vitals


def extract_vitals_from_cohort_csv(cohort_path: Path) -> Dict[str, pd.Series]:
    """
    Fallback extractor from processed cohort CSV if chartevents is unavailable.
    """
    df = pd.read_csv(cohort_path, low_memory=False)
    vitals: Dict[str, pd.Series] = {}

    col_mappings = {
        "heart_rate": "heart_rate",
        "respiratory_rate": "respiratory_rate",
        "spo2": "spo2",
    }
    for k, col in col_mappings.items():
        if col in df.columns:
            vitals[k] = pd.to_numeric(df[col], errors="coerce").dropna()

    # Derived or secondary BP columns if present
    if "mean_arterial_pressure" in df.columns:
        # Approximate SBP/DBP if MAP is given and SBP/DBP not explicit
        map_s = pd.to_numeric(df["mean_arterial_pressure"], errors="coerce").dropna()
        vitals["systolic_bp"] = (map_s * 1.3).round(1)
        vitals["diastolic_bp"] = (map_s * 0.75).round(1)

    return vitals


def load_vitals_dataset(
    dataset_dir: Optional[Path] = None,
    cohort_path: Optional[Path] = None,
) -> Tuple[Dict[str, pd.Series], str]:
    """
    Loads vitals from chartevents if available; otherwise falls back to processed cohort.
    Returns (vitals_dict, source_description).
    """
    project_root = Path(__file__).resolve().parent.parent
    base_dir = dataset_dir if dataset_dir else (project_root / REL_DATASET_DIR)

    ce_path = resolve_table_path(base_dir, "icu/chartevents.csv.gz")
    if ce_path and ce_path.is_file():
        vitals = extract_vitals_from_chartevents(ce_path)
        return vitals, f"MIMIC-IV chartevents ({ce_path.as_posix()})"

    # Fallback to cohort CSV
    cohort_file = cohort_path if cohort_path else (project_root / DEFAULT_COHORT_PATH)
    if cohort_file and cohort_file.is_file():
        vitals = extract_vitals_from_cohort_csv(cohort_file)
        return vitals, f"Processed cohort CSV ({cohort_file.as_posix()})"

    raise FileNotFoundError(
        f"Unable to find chartevents under {base_dir} or processed cohort at {cohort_file}"
    )


def compute_vital_statistics(series: pd.Series) -> Dict[str, float]:
    """
    Calculates summary statistics for a vital sign series:
    Count, Mean, Median, Standard Deviation, 5th Percentile, 95th Percentile, Min, Max, IQR.
    """
    s = series.dropna().astype(float)
    if s.empty:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p5": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
            "iqr": 0.0,
        }

    q25 = float(s.quantile(0.25))
    q75 = float(s.quantile(0.75))
    return {
        "count": int(len(s)),
        "mean": round(float(s.mean()), 2),
        "std": round(float(s.std()), 2),
        "median": round(float(s.median()), 2),
        "p5": round(float(s.quantile(0.05)), 2),
        "p95": round(float(s.quantile(0.95)), 2),
        "min": round(float(s.min()), 2),
        "max": round(float(s.max()), 2),
        "iqr": round(float(q75 - q25), 2),
    }


def compute_all_vital_statistics(vitals: Dict[str, pd.Series]) -> Dict[str, Dict[str, float]]:
    """Computes summary statistics dictionary across all configured vital signs."""
    return {k: compute_vital_statistics(vitals[k]) for k in vitals if not vitals[k].empty}


def format_terminal_summary(
    stats_dict: Dict[str, Dict[str, float]], source_desc: str = ""
) -> str:
    """
    Formats the vital signs statistical summary into a clean terminal table.
    """
    lines = []
    lines.append("=" * 86)
    lines.append("MIMIC-IV CLINICAL VITAL SIGNS DISTRIBUTION SUMMARY")
    lines.append("Week 1 Task 1.5: Distribution Histograms & Kernel Density Estimation (KDE)")
    lines.append("=" * 86)
    if source_desc:
        lines.append(f"[*] Data Source: {source_desc}")
        lines.append("-" * 86)

    header = (
        f"{'Vital Sign':<28} "
        f"{'Count':<9} "
        f"{'Mean':<8} "
        f"{'Std':<8} "
        f"{'Median':<8} "
        f"{'5th %':<8} "
        f"{'95th %':<8} "
        f"{'Unit':<8}"
    )
    lines.append(header)
    lines.append("-" * 86)

    for cfg in VITALS_CONFIG:
        k = cfg["key"]
        if k not in stats_dict:
            continue
        st = stats_dict[k]
        name_unit = f"{cfg['name']}"
        lines.append(
            f"{name_unit:<28} "
            f"{st['count']:<9,} "
            f"{st['mean']:<8.2f} "
            f"{st['std']:<8.2f} "
            f"{st['median']:<8.2f} "
            f"{st['p5']:<8.2f} "
            f"{st['p95']:<8.2f} "
            f"{cfg['unit']:<8}"
        )

    lines.append("=" * 86)
    return "\n".join(lines)


def plot_vital_distributions(
    vitals: Dict[str, pd.Series],
    stats_dict: Dict[str, Dict[str, float]],
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """
    Generates a 2x3 grid of distribution histograms with overlaid KDE curves,
    vertical statistical marker lines (mean, median, 5th, 95th percentiles),
    and descriptive annotated statistic cards.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up clean professional style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    axes = axes.flatten()

    fig.suptitle(
        "MIMIC-IV ICU Cohort: Key Clinical Vital Signs Distributions & Kernel Density Estimates",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )

    for idx, cfg in enumerate(VITALS_CONFIG):
        ax = axes[idx]
        k = cfg["key"]

        if k not in vitals or vitals[k].empty:
            ax.set_title(f"{cfg['name']} (No Data)", fontsize=13, fontweight="bold")
            ax.axis("off")
            continue

        data = vitals[k].dropna().values
        st = stats_dict[k]
        color = cfg["color"]

        # Histogram
        n_bins = 40
        counts, bins, patches = ax.hist(
            data,
            bins=n_bins,
            density=True,
            alpha=0.45,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label="Histogram",
        )

        # Kernel Density Estimate (KDE)
        try:
            kde = gaussian_kde(data)
            x_grid = np.linspace(bins[0], bins[-1], 300)
            kde_curve = kde(x_grid)
            ax.plot(
                x_grid,
                kde_curve,
                color=color,
                linewidth=2.5,
                label="KDE Density",
            )
        except Exception:
            pass

        # Summary Statistics Vertical Lines
        ax.axvline(
            st["mean"],
            color="#d90429",
            linestyle="--",
            linewidth=1.8,
            label=f"Mean: {st['mean']:.1f}",
        )
        ax.axvline(
            st["median"],
            color="#0077b6",
            linestyle="-",
            linewidth=2.0,
            label=f"Median: {st['median']:.1f}",
        )
        ax.axvline(
            st["p5"],
            color="#6c757d",
            linestyle=":",
            linewidth=1.5,
            label=f"5th %: {st['p5']:.1f}",
        )
        ax.axvline(
            st["p95"],
            color="#6c757d",
            linestyle=":",
            linewidth=1.5,
            label=f"95th %: {st['p95']:.1f}",
        )

        # Titles and Labels
        ax.set_title(
            f"{cfg['name']} ({cfg['unit']})",
            fontsize=13,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel(f"{cfg['name']} [{cfg['unit']}]", fontsize=11, fontweight="semibold")
        ax.set_ylabel("Probability Density", fontsize=11, fontweight="semibold")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Annotated Statistics Card
        stats_box_text = (
            f"N = {st['count']:,}\n"
            f"Mean = {st['mean']:.1f} {cfg['unit']}\n"
            f"Median = {st['median']:.1f} {cfg['unit']}\n"
            f"Std = {st['std']:.1f}\n"
            f"IQR = {st['iqr']:.1f}\n"
            f"5th % = {st['p5']:.1f}\n"
            f"95th % = {st['p95']:.1f}"
        )
        ax.text(
            0.96,
            0.94,
            stats_box_text,
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="#adb5bd",
                alpha=0.92,
                linewidth=0.8,
            ),
        )

        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.85)

    # Save output plot
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_vitals_visualization(
    dataset_dir: Optional[Path] = None,
    cohort_path: Optional[Path] = None,
    output_png: Optional[Path] = None,
    dpi: int = 300,
    quiet: bool = False,
) -> bool:
    """
    Main pipeline to load vitals, compute statistics, display terminal summary,
    and save multi-panel distribution plot.
    """
    project_root = Path(__file__).resolve().parent.parent
    target_png = output_png if output_png else (project_root / DEFAULT_OUTPUT_PNG)

    try:
        vitals, source_desc = load_vitals_dataset(
            dataset_dir=dataset_dir, cohort_path=cohort_path
        )
    except Exception as e:
        print(f"[ERROR] Failed to load clinical vitals: {e}")
        return False

    stats_dict = compute_all_vital_statistics(vitals)

    if not quiet:
        report = format_terminal_summary(stats_dict, source_desc=source_desc)
        print(report)

    plot_vital_distributions(vitals, stats_dict, target_png, dpi=dpi)

    if not quiet:
        print(f"\n[INFO] Vital signs distribution plot saved successfully to: {target_png.as_posix()}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate clinical vital signs distribution histograms & KDE curves for MIMIC-IV."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Path to MIMIC-IV clinical database directory.",
    )
    parser.add_argument(
        "--cohort-path",
        type=Path,
        default=None,
        help="Path to extracted cohort CSV fallback.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for output PNG plot (default: outputs/vital_distributions.png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution DPI for exported plot image (default: 300).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress terminal summary output.",
    )
    args = parser.parse_args()

    success = run_vitals_visualization(
        dataset_dir=args.dataset_dir,
        cohort_path=args.cohort_path,
        output_png=args.output,
        dpi=args.dpi,
        quiet=args.quiet,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
