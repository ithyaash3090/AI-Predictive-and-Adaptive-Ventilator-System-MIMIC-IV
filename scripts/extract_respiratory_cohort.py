"""
Respiratory Cohort Extractor for MIMIC-IV Ventilator System.
Filters the general patient cohort to extract patients with confirmed
respiratory-oriented diseases (Acute Respiratory Failure, ARDS, Severe Pneumonia,
COPD Exacerbation, and Hypoxemic Respiratory Distress).
Outputs: outputs/respiratory_cohort.csv
"""

import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard project paths
DATASET_DIR = Path("mimic-iv-demo-dataset/mimic-iv-clinical-database-demo-2.2")
PROCESSED_COHORT_PATH = Path("outputs/processed_mimic_cohort.csv")
OUTPUT_RESPIRATORY_PATH = Path("outputs/respiratory_cohort.csv")

# Standard ICD terminology for Respiratory Pathology
RESPIRATORY_ICD_KEYWORDS = [
    "acute respiratory failure",
    "respiratory failure",
    "respiratory distress",
    "ards",
    "pneumonia",
    "chronic obstructive",
    "copd",
    "hypoxemia",
    "ventilator",
    "emphysema",
    "asthma",
    "pulmonary edema",
    "aspiration"
]


def classify_primary_respiratory_disease(icd_title: str) -> str:
    """Categorizes long medical diagnosis title into primary respiratory etiology."""
    t = str(icd_title).lower()
    if "acute respiratory failure" in t or "respiratory failure with hypoxia" in t:
        return "Acute Respiratory Failure"
    elif "ards" in t or "acute respiratory distress" in t:
        return "ARDS"
    elif "pneumonia" in t:
        return "Severe Pneumonia"
    elif "chronic obstructive" in t or "copd" in t or "emphysema" in t:
        return "COPD Exacerbation"
    elif "asthma" in t or "broncho" in t:
        return "Asthma / Bronchospasm"
    elif "hypoxemia" in t or "pulmonary edema" in t:
        return "Acute Hypoxemic Distress"
    return "Secondary Pulmonary Compromise"


def extract_respiratory_cohort():
    logger.info("Extracting respiratory-oriented cohort from MIMIC-IV...")

    # 1. Load diagnoses tables
    diag_path = DATASET_DIR / "hosp/diagnoses_icd.csv/diagnoses_icd.csv"
    if not diag_path.exists():
        diag_path = DATASET_DIR / "hosp/diagnoses_icd.csv.gz"
    
    d_diag_path = DATASET_DIR / "hosp/d_icd_diagnoses.csv/d_icd_diagnoses.csv"
    if not d_diag_path.exists():
        d_diag_path = DATASET_DIR / "hosp/d_icd_diagnoses.csv.gz"

    diag = pd.read_csv(diag_path)
    d_diag = pd.read_csv(d_diag_path)

    # 2. Filter respiratory terms in ICD dictionary
    pattern = "|".join(RESPIRATORY_ICD_KEYWORDS)
    resp_dictionary = d_diag[d_diag["long_title"].str.contains(pattern, case=False, na=False)].copy()
    resp_dictionary["respiratory_category"] = resp_dictionary["long_title"].apply(classify_primary_respiratory_disease)

    # 3. Match with patient diagnosis records
    patient_resp_diag = diag.merge(resp_dictionary, on=["icd_code", "icd_version"])
    logger.info(f"Identified {len(patient_resp_diag)} diagnosis records matching respiratory disease.")

    # 4. Map primary etiology per admission (hadm_id)
    # Priority order for primary diagnosis
    priority_order = [
        "Acute Respiratory Failure",
        "ARDS",
        "Severe Pneumonia",
        "COPD Exacerbation",
        "Acute Hypoxemic Distress",
        "Asthma / Bronchospasm",
        "Secondary Pulmonary Compromise"
    ]
    
    admission_primary = {}
    for hadm_id, group in patient_resp_diag.groupby("hadm_id"):
        # Select highest priority category
        assigned = None
        for p in priority_order:
            if p in group["respiratory_category"].values:
                assigned = p
                break
        admission_primary[hadm_id] = assigned or group["respiratory_category"].iloc[0]

    resp_subjects = set(patient_resp_diag["subject_id"])
    logger.info(f"Unique respiratory patients identified: {len(resp_subjects)}")

    # 5. Load processed time-series cohort
    if not PROCESSED_COHORT_PATH.exists():
        raise FileNotFoundError(f"Processed cohort file not found: {PROCESSED_COHORT_PATH}")

    df_processed = pd.read_csv(PROCESSED_COHORT_PATH)
    logger.info(f"Loaded processed cohort: {len(df_processed)} rows across {df_processed['stay_id'].nunique()} stays.")

    # 6. Filter processed rows for respiratory patients / stays
    resp_cohort = df_processed[df_processed["subject_id"].isin(resp_subjects)].copy()
    
    # Add explicit respiratory disease label column
    resp_cohort["primary_respiratory_etiology"] = resp_cohort["hadm_id"].map(admission_primary).fillna("Acute Respiratory Distress")
    resp_cohort["is_respiratory_cohort"] = 1

    # 7. Persist to outputs/respiratory_cohort.csv
    OUTPUT_RESPIRATORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    resp_cohort.to_csv(OUTPUT_RESPIRATORY_PATH, index=False)

    logger.info(f"Successfully saved respiratory cohort to {OUTPUT_RESPIRATORY_PATH}")
    logger.info("=" * 65)
    logger.info(f"Respiratory Cohort Summary:")
    logger.info(f"  - Total Hourly Time-Series Rows : {len(resp_cohort):,} ({len(resp_cohort)/len(df_processed)*100:.1f}% of cohort)")
    logger.info(f"  - Unique Respiratory Patients   : {resp_cohort['subject_id'].nunique()}")
    logger.info(f"  - Unique ICU Stays              : {resp_cohort['stay_id'].nunique()}")
    logger.info(f"  - Positive Deterioration Rate   : {resp_cohort['target_deterioration_lead'].mean()*100:.2f}%")
    logger.info(f"  - Primary Etiology Breakdown    :")
    for cat, count in resp_cohort.groupby("primary_respiratory_etiology")["stay_id"].nunique().items():
        logger.info(f"      * {cat:30} : {count} stays")
    logger.info("=" * 65)

    return resp_cohort


if __name__ == "__main__":
    extract_respiratory_cohort()
