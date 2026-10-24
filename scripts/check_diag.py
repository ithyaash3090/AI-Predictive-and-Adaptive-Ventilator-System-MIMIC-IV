import pandas as pd

hosp_dir = 'mimic-iv-demo-dataset/mimic-iv-clinical-database-demo-2.2/hosp'

diag = pd.read_csv(f'{hosp_dir}/diagnoses_icd.csv/diagnoses_icd.csv')
d_diag = pd.read_csv(f'{hosp_dir}/d_icd_diagnoses.csv/d_icd_diagnoses.csv')

diag_merged = diag.merge(d_diag, on=['icd_code', 'icd_version'], how='left')

resp_keywords = ['respiratory failure', 'ards', 'pneumonia', 'copd', 'hypox', 'sepsis']
pattern = '|'.join(resp_keywords)

matched_diag = diag_merged[diag_merged['long_title'].str.lower().str.contains(pattern, na=False)]
print(f"Total respiratory / sepsis diagnosis records: {len(matched_diag)}")
print(matched_diag['long_title'].value_counts().head(15))
print(f"Unique subject_ids with respiratory / sepsis diagnosis: {matched_diag['subject_id'].nunique()}")
