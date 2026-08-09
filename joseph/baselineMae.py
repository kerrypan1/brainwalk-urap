import numpy as np
import pandas as pd
import os

from fgaFinding import SplitManagerDataset, FEATURE_CSV, SPLIT_CSV
from zenoCombinedModel import TARGET_FEATURES

ZENO_SCORE_PATH = "[redacted]"

def get_bath_baselines():
    metrics = [
        "imbalance_label", 
        "speed_label", 
        "lateral_deviation_label", 
        "gait_deviation_label",
        "device_label",
        "fga_label"
    ]
    
    results = {}

    print(f"{'Bath Metric':<25} | {'Baseline MAE':<15}")
    print("-" * 45)

    for metric in metrics:
        ds = SplitManagerDataset(FEATURE_CSV, SPLIT_CSV, target_metric=metric) #load data
        
        fold_maes = []
        for fold in range(5):
            train_idx = np.where(ds.fold_assignments != fold)[0] #should match the folds
            val_idx = np.where(ds.fold_assignments == fold)[0]
            
            baseline_guess = ds.labels[train_idx].mean() #what we're gonna be using
            
            actual_values = ds.labels[val_idx]
            mae = np.mean(np.abs(actual_values - baseline_guess)) #comparisons
            fold_maes.append(mae)
            
        results[metric] = np.mean(fold_maes)
        print(f"{metric:<25} | {results[metric]:.4f}")
    print("-" * 45)

def calculate_zeno_baselines():
    split_df = pd.read_csv(SPLIT_CSV) #load
    zeno_df = pd.read_excel(ZENO_SCORE_PATH)
    
    #merge so split correct
    master_df = pd.merge(zeno_df, split_df[['participant_id', 'split']], 
                         left_on='bw_id', right_on='participant_id')
    
    all_feature_baselines = [] #set up for comparison 
    
    print(f"{'Zeno Metric':<25} | {'Baseline MAE':<15}")
    print("-" * 45)
    
    for feature in TARGET_FEATURES:
        actual_col = next((c for c in master_df.columns if c.lower() == feature.lower()), None) #make sure no errors if non existing column
        
        if not actual_col:
            continue
            
        fold_maes = []
        for fold_val in range(5):
            fold_name = f"fold_{fold_val}" 
            
            train_data = master_df[master_df['split'] != fold_name][actual_col].dropna()
            val_data = master_df[master_df['split'] == fold_name][actual_col].dropna()
            
            if len(train_data) > 0 and len(val_data) > 0:
                train_mean = train_data.mean()
                mae = np.mean(np.abs(val_data - train_mean))
                fold_maes.append(mae)
                
            
        if fold_maes: #to ensure not empty
            avg_mae = np.mean(fold_maes)
            all_feature_baselines.append(avg_mae)
            print(f"{feature:<25} | {avg_mae:.4f}")

    if all_feature_baselines: #this is just at the end so can do the comparable
        combined_baseline = np.mean(all_feature_baselines)
        print("-" * 45)
        print(f"{'COMBINED ZENO BASELINE':<25} | {combined_baseline:.4f}")

if __name__ == "__main__":
    get_bath_baselines()
    calculate_zeno_baselines()
