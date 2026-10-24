import pandas as pd

icu_dir = 'mimic-iv-demo-dataset/mimic-iv-clinical-database-demo-2.2/icu'
hosp_dir = 'mimic-iv-demo-dataset/mimic-iv-clinical-database-demo-2.2/hosp'

items_of_interest = {
    220045: 'Heart Rate',
    220210: 'Respiratory Rate',
    220277: 'SpO2',
    220052: 'Arterial BP Mean',
    220181: 'NBP Mean',
    223835: 'FiO2',
    220339: 'PEEP Set',
    224700: 'PEEP Total',
    224684: 'Tidal Vol Set',
    224685: 'Tidal Vol Observed',
    224695: 'Peak Insp Pressure',
    224696: 'Plateau Pressure',
    224687: 'Minute Volume',
    223849: 'Ventilator Mode',
}

# chartevents is ~64MB, we can read chunks or load selected columns
chunks = []
print("Reading chartevents for selected items...")
for chunk in pd.read_csv(f'{icu_dir}/chartevents.csv/chartevents.csv', 
                         usecols=['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'valuenum', 'value', 'valueuom'],
                         chunksize=100000):
    m = chunk[chunk['itemid'].isin(items_of_interest.keys())]
    if len(m) > 0:
        chunks.append(m)

df_chart = pd.concat(chunks, ignore_index=True)
print(f"Total matching records in chartevents: {len(df_chart)}")
df_chart['item_name'] = df_chart['itemid'].map(items_of_interest)
print(df_chart['item_name'].value_counts())
print("Unique stay_ids with these chart items:", df_chart['stay_id'].nunique())
print("Unique subject_ids:", df_chart['subject_id'].nunique())

# Check labevents
lab_items = {
    50821: 'pO2',
    50818: 'pCO2',
    50820: 'pH',
    50813: 'Lactate',
}
lab_chunks = []
for chunk in pd.read_csv(f'{hosp_dir}/labevents.csv/labevents.csv',
                         usecols=['labevent_id', 'subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum', 'valueuom'],
                         chunksize=100000):
    m = chunk[chunk['itemid'].isin(lab_items.keys())]
    if len(m) > 0:
        lab_chunks.append(m)

df_lab = pd.concat(lab_chunks, ignore_index=True)
print(f"\nTotal matching records in labevents: {len(df_lab)}")
df_lab['lab_name'] = df_lab['itemid'].map(lab_items)
print(df_lab['lab_name'].value_counts())
print("Unique subject_ids with blood gas:", df_lab['subject_id'].nunique())
