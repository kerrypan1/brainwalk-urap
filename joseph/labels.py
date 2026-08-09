import os
import re
import numpy as np
import pandas as pd

GAIT_FEATURE_DIR = "[redacted]"
VIDEO_XLSX = "[redacted].xlsx"
OUTPUT_CSV = "[redacted].csv"

def parse_pose_filename(fname): #lets us match the numpy ffile to the partipant in the bath spreadsheet
    base = os.path.basename(fname).replace(".npy", "")
    parts = base.split("_")

    bw_id = parts[0]                         
    date = parts[1]                        
    visit_date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    
    # task = parts[2] # unused
    cam = parts[3] 
    video_idx = 1 if cam == "cam1" else 2

    return {
        "bw_id": bw_id,
        "visit_date": visit_date,
        "video_idx": video_idx,
        "pose_path": os.path.join(GAIT_FEATURE_DIR, fname)
    }

def parse_clinical_score(x): #basically parses the first number (score we want) and ignores any commends 
    if pd.isna(x):
        return None
    
    #regex: ^(\d+) looks for digits at the very start of the string
    match = re.match(r"\s*(\d+)", str(x))
    if match:
        return int(match.group(1))
    return None

pose_rows = [] #pose list
if os.path.exists(GAIT_FEATURE_DIR):
    for f in os.listdir(GAIT_FEATURE_DIR):
        if f.endswith("_gait.npy"):
            pose_rows.append(parse_pose_filename(f))

pose_df = pd.DataFrame(pose_rows)
print(f"Found {len(pose_df)} pose files")

#excel labels
df = pd.read_excel(VIDEO_XLSX, header=1) 

new_columns = [] #as notes inherently part of column so annoying to do straight
for col in df.columns:
    col_str = str(col).strip()
    
    if "speed1" in col_str:
        new_columns.append("speed1")  #rename messy string to "speed1"
    elif "speed2" in col_str:
        new_columns.append("speed2")  #rename messy string to "speed2"
    else:
        new_columns.append(col_str)   #keep others (like imbalance1) as is

df.columns = new_columns
df["bw_id"] = df["BW-ID"].astype(str).str.strip()

#map cols
rows = []
video_map = {
    1: {
        "date": "visit_date_video1",
        "imbalance": "imbalance1",
        "speed": "speed1",                  
        "gait_deviation": "gait_deviation1",
        "device": "assistive_device1",
        "fga": "FGA_estimate_score1",
        "lateral_dev": "deviation_outside_walkway1"
    },
    2: {
        "date": "visit_date_video2",
        "imbalance": "imbalance2",
        "speed": "speed2",                   
        "gait_deviation": "gait_deviation2",
        "device": "assistive_device2",
        "fga": "FGA_estimate_score2",
        "lateral_dev": "deviation_outside_walkway2"
    }
}

for _, r in df.iterrows():
    for vidx, cols in video_map.items():
        # Only process if there is a valid date for this specific video entry
        if cols["date"] in r and pd.notna(r[cols["date"]]):
            
            #extract everything
            imb = parse_clinical_score(r.get(cols["imbalance"]))
            spd = parse_clinical_score(r.get(cols["speed"]))
            dev = parse_clinical_score(r.get(cols["gait_deviation"]))
            dev_type = parse_clinical_score(r.get(cols["device"]))
            fga = parse_clinical_score(r.get(cols["fga"]))
            lat_dev = parse_clinical_score(r.get(cols["lateral_dev"]))

            #put all in rows 
            rows.append({
                "bw_id": r["bw_id"],
                "visit_date": pd.to_datetime(r[cols["date"]], errors="coerce").strftime("%Y-%m-%d"),
                "video_idx": vidx,
                "imbalance_label": imb,
                "speed_label": spd,         
                "gait_deviation_label": dev,
                "device_label": dev_type,
                "fga_label": fga,
                "lateral_deviation_label": lat_dev
            })

label_df = pd.DataFrame(rows)

label_df = label_df.dropna(subset=["imbalance_label", "speed_label", "gait_deviation_label"], how="all") #if all 3 missing drop the row -> carry over from when only using these 3 but still good indicator
print(f"Found {len(label_df)} labeled videos")

#merge poses and labels together
dataset_df = pose_df.merge(
    label_df,
    on=["bw_id", "visit_date", "video_idx"],
    how="inner"
)

print(f"Final training samples: {len(dataset_df)}")
dataset_df.to_csv(OUTPUT_CSV, index=False)
