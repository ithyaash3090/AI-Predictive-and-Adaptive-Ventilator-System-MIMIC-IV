"""
Run extraction pipeline on MIMIC-IV demo dataset and save processed time-series.
"""
import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath('.'))

from src.data.extractor import MimicIvExtractor

def main():
    extractor = MimicIvExtractor()
    cohort = extractor.load_cohort()
    df_diag = extractor.load_comorbidities()
    
    stay_ids = cohort['stay_id'].unique()
    subject_ids = cohort['subject_id'].unique()
    
    df_chart = extractor.extract_chartevents(stay_ids)
    df_lab = extractor.extract_labevents(subject_ids)
    
    df_ts = extractor.build_hourly_timeseries(cohort, df_chart, df_lab, df_diag)
    df_labeled = extractor.label_respiratory_deterioration(df_ts, lookahead_hours=4)
    
    output_path = "outputs/processed_mimic_cohort.csv"
    df_labeled.to_csv(output_path, index=False)
    print(f"\n>>> Saved processed MIMIC-IV cohort dataset to {output_path}")
    print(f"Shape: {df_labeled.shape}")
    print("Columns:", list(df_labeled.columns))
    print("\nTarget Class Distribution:")
    print(df_labeled['target_deterioration_lead'].value_counts(normalize=True))

if __name__ == "__main__":
    main()
