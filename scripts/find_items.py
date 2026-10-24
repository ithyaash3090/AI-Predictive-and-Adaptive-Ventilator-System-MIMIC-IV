import pandas as pd

icu_dir = 'mimic-iv-demo-dataset/mimic-iv-clinical-database-demo-2.2/icu'
d_items = pd.read_csv(f'{icu_dir}/d_items.csv/d_items.csv')

terms = ['inspired o2 fraction', 'fio2', 'spo2', 'o2 saturation', 'heart rate', 'mean airway pressure', 'peak inspiratory pressure', 'pip', 'peep', 'tidal volume', 'minute volume', 'blood pressure mean', 'temperature']

for term in terms:
    m = d_items[d_items['label'].str.lower().str.contains(term, na=False)]
    print(f"=== Term: '{term}' ({len(m)} matches) ===")
    for _, r in m[['itemid', 'label', 'category', 'unitname']].head(5).iterrows():
        print(f"  {r['itemid']}: {r['label']} | cat: {r['category']} | unit: {r['unitname']}")
