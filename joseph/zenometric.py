import pandas as pd
import numpy as np
import os
import json

INPUT_FILE = "[redacted].xlsx"
OUTPUT_DIR = os.path.expanduser("[redacted]")
os.makedirs(OUTPUT_DIR, exist_ok=True)

id_col = 'bw_id'
date_col = 'trialdate_zeno' #use Zeno specific recording date

df = pd.read_excel(INPUT_FILE, na_values=["Not Calculated", "n/a", " "]) #multiple ways something could be null
df = df.dropna(subset=[id_col]) #if no id then don't want

if 'gaitProtocol' in df.columns:
    df = df[df['gaitProtocol'] == 'PWS'].copy() #would need to change this to 'FW' as supposedly this matches the Bath scores but for now is fine
    print(f"Filtered PWS only: {len(df)} rows remaining")
else:
    print("WARNING: 'gaitProtocol' not found in Excel file!")

for col in df.columns: #due to strings is obj so want floats
    if col not in [id_col, date_col, 'visit_date_demo']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

initial_features = df.select_dtypes(include=[np.number]).columns.tolist() #just getting numerical
initial_features = [f for f in initial_features if f not in [id_col, date_col, 'visit_date_demo']] #exclude stuff we already have

col_thresh = len(df) * 0.70 #any higher only get 3 features
df_balanced = df.dropna(thresh=col_thresh, axis=1)

features_balanced = [f for f in initial_features if f in df_balanced.columns]
df_balanced.loc[:, features_balanced] = df_balanced[features_balanced].fillna(df_balanced[features_balanced].median()) #fill nulls
df_final = df_balanced.dropna(subset=features_balanced, axis=0)

print(f" - Features retained: {len(features_balanced)}")
print(f" - Rows preserved: {len(df_final)}")
print(f" - Total NaNs: {df_final[features_balanced].isna().sum().sum()}")

count = 0
for patient_id, group in df_final.groupby(id_col):
    patient_data = group.sort_values(by=date_col)[features_balanced].values
    
    file_path = os.path.join(OUTPUT_DIR, f"{str(patient_id)}.npy")
    np.save(file_path, patient_data)
    count += 1

print(f"Saved {count} files.")

names_path = os.path.join(OUTPUT_DIR, "feature_names.json") #for feature selection -> well no longer using to match Qixing but will keep
with open(names_path, "w") as f:
    json.dump(features_balanced, f)
print(f"Saved feature names to {names_path}")
