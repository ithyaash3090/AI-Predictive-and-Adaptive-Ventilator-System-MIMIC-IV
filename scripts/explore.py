import pandas as pd
import os

icu_dir = 'mimic-iv-demo-dataset/mimic-iv-clinical-database-demo-2.2/icu'
hosp_dir = 'mimic-iv-demo-dataset/mimic-iv-clinical-database-demo-2.2/hosp'

d_items = pd.read_csv(f'{icu_dir}/d_items.csv/d_items.csv')
print('d_items shape:', d_items.shape)

vent_keywords = ['peep', 'fio2', 'tidal', 'ventilator', 'plateau', 'peak inspiratory', 'minute volume', 'spo2', 'respiratory rate']
matches = d_items[d_items['label'].str.lower().str.contains('|'.join(vent_keywords), na=False)]
print(f'Matched items in d_items ({len(matches)}):')
for _, row in matches[['itemid', 'label', 'abbreviation', 'category']].head(35).iterrows():
    print(f"  {row['itemid']}: {row['label']} ({row['category']})")

icustays = pd.read_csv(f'{icu_dir}/icustays.csv/icustays.csv')
print('\nTotal icustays in demo:', len(icustays), 'Unique subject_ids:', icustays['subject_id'].nunique())

patients = pd.read_csv(f'{hosp_dir}/patients.csv/patients.csv')
print('Patients shape:', patients.shape)

admissions = pd.read_csv(f'{hosp_dir}/admissions.csv/admissions.csv')
print('Admissions shape:', admissions.shape)

procedureevents = pd.read_csv(f'{icu_dir}/procedureevents.csv/procedureevents.csv')
print('Procedureevents items:', procedureevents['itemid'].value_counts())
proc_matches = d_items[d_items['itemid'].isin(procedureevents['itemid'].unique())]
for _, row in proc_matches.iterrows():
    print(f"  Proc Item {row['itemid']}: {row['label']}")
