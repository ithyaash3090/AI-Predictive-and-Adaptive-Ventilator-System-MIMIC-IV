import pandas as pd

icu_dir = 'mimic-iv-demo-dataset/mimic-iv-clinical-database-demo-2.2/icu'
hosp_dir = 'mimic-iv-demo-dataset/mimic-iv-clinical-database-demo-2.2/hosp'

d_items = pd.read_csv(f'{icu_dir}/d_items.csv/d_items.csv')
d_labs = pd.read_csv(f'{hosp_dir}/d_labitems.csv/d_labitems.csv')

print("--- Peak pressure ---")
m = d_items[d_items['label'].str.lower().str.contains('peak', na=False)]
for _, r in m[['itemid', 'label', 'category', 'unitname']].iterrows():
    print(f"  {r['itemid']}: {r['label']} | cat: {r['category']} | unit: {r['unitname']}")

print("--- Labs (PaO2, PaCO2, pH, Lactate) ---")
lab_terms = ['po2', 'pco2', 'ph', 'lactate', 'bicarbonate', 'base excess']
for lt in lab_terms:
    ml = d_labs[d_labs['label'].str.lower().str.contains(lt, na=False)]
    for _, r in ml[['itemid', 'label', 'category', 'fluid']].head(3).iterrows():
        print(f"  {r['itemid']}: {r['label']} | cat: {r['category']} | fluid: {r['fluid']}")
