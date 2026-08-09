import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LinearRegression
from fgaFinding import SplitManagerDataset, FEATURE_CSV, SPLIT_CSV, new_collate_fn
from multiple import DEVICE, set_seed, RANDOM_STATE, MetricLSTM
from combined import FGA_Estimator

KERRY_BOOTSTRAPS = np.load("[redacted].npz", allow_pickle=True) #for dicts 

def get_kerry_indices(fold_idx):
    if f"fold_{fold_idx}" in KERRY_BOOTSTRAPS: #sometimes inconsistent naming
        return KERRY_BOOTSTRAPS[f"fold_{fold_idx}"]
    elif f"fold_{fold_idx}.npy" in KERRY_BOOTSTRAPS:
        return KERRY_BOOTSTRAPS[f"fold_{fold_idx}.npy"]
    else:
        raise KeyError(f"Could not find fold_{fold_idx}, Available keys: {KERRY_BOOTSTRAPS.files}")

def apply_kerry_bootstraps(fold_gt_dict, fold_preds_dict, fold_idx_map, label_name):
    all_folds_mae_dist = [] #for the average
    all_folds_mse_dist = []
    
    for fold in range(5): #5 folds
        gt = fold_gt_dict[fold] #ground truth 
        preds = fold_preds_dict[fold]
        kerry_indices = get_kerry_indices(fold) 
        
        global_to_local = {int(csv_idx): i for i, csv_idx in enumerate(fold_idx_map[fold])} #need to map full spreadsheet indices to the kerry ones 
        
        fold_mae_samples = [] #for this specific fold
        fold_mse_samples = []

        for sample_indices in kerry_indices: #only include indices that exist -> no key error if filtered out patient with NaN
            valid_indices = [global_to_local[idx] for idx in sample_indices if idx in global_to_local]
            
            if len(valid_indices) == 0:
                continue
                
            valid_indices = np.array(valid_indices)
            
            curr_mae = np.mean(np.abs(preds[valid_indices] - gt[valid_indices]))
            curr_mse = np.mean((preds[valid_indices] - gt[valid_indices])**2)
            
            fold_mae_samples.append(curr_mae)
            fold_mse_samples.append(curr_mse)
            
        all_folds_mae_dist.extend(fold_mae_samples)
        all_folds_mse_dist.extend(fold_mse_samples)
    
    return {
        "metric": label_name,
        "mae": np.mean(all_folds_mae_dist),
        "mae_ci": (np.percentile(all_folds_mae_dist, 2.5), np.percentile(all_folds_mae_dist, 97.5)),
        "mse": np.mean(all_folds_mse_dist),
        "mse_ci": (np.percentile(all_folds_mse_dist, 2.5), np.percentile(all_folds_mse_dist, 97.5)),
    }

def get_fold_predictions(metric_name):
    bath_components = ["imbalance_label", "speed_label", "lateral_deviation_label", "gait_deviation_label", "device_label"]
    if metric_name in bath_components: target_col = metric_name
    else: target_col = "fga_label" #so for like combined 
    
    ds = SplitManagerDataset(FEATURE_CSV, SPLIT_CSV, target_metric=target_col)
    split_df = pd.read_csv(SPLIT_CSV)
    pid_to_csv_idx = {pid: idx for idx, pid in enumerate(split_df['participant_id'])} #mapping BW_[] to the split row (so can match fga)
    
    fold_gt, fold_preds, fold_idx_map = {}, {}, {}

    #fusion -> retrain as just linear regression so literally 2 weights
    if metric_name == "fusion_fga_label":
        zeno_df = pd.read_csv("zeno_predictions.csv")
        video_df = pd.read_csv("video_predictions.csv")
        video_avg = video_df.groupby(['participant_id', 'fold'])[['video_pred', 'actual_fga']].mean().reset_index()
        fusion_df = pd.merge(zeno_df[['participant_id', 'fold', 'zeno_pred']], video_avg, on=["participant_id", "fold"])

        for fold in range(5):
            train_set = fusion_df[fusion_df['fold'] != fold]
            test_set = fusion_df[fusion_df['fold'] == fold].copy()
            meta_model = LinearRegression().fit(train_set[['zeno_pred', 'video_pred']], train_set['actual_fga'])
            fold_gt[fold] = test_set['actual_fga'].values
            fold_preds[fold] = meta_model.predict(test_set[['zeno_pred', 'video_pred']])
            fold_idx_map[fold] = test_set['participant_id'].map(pid_to_csv_idx).values
        return fold_gt, fold_preds, fold_idx_map

    #zeno combined (read from csv)
    elif metric_name == "combined_zeno_label":
        zeno_df = pd.read_csv("zeno_predictions.csv")
        for fold in range(5):
            fold_df = zeno_df[zeno_df['fold'] == fold]
            fold_gt[fold] = fold_df['actual_fga'].values
            fold_preds[fold] = fold_df['zeno_pred'].values
            fold_idx_map[fold] = fold_df['participant_id'].map(pid_to_csv_idx).values
        return fold_gt, fold_preds, fold_idx_map

    #bath metrcic including combined
    else:
        for fold in range(5):
            val_idx = np.where(ds.fold_assignments == fold)[0]
            val_loader = DataLoader(Subset(ds, val_idx), batch_size=16, shuffle=False, collate_fn=new_collate_fn)
            
            if metric_name == "combined_bath_label": #basically just load the weights in for that fold and evaluate what the result of the model is
                model = FGA_Estimator(
                    f"imbalance_label_fold{fold}.pth", f"speed_label_fold{fold}.pth",
                    f"lateral_deviation_label_fold{fold}.pth", f"gait_deviation_label_fold{fold}.pth",
                    f"device_label_fold{fold}.pth"
                ).to(DEVICE)
                model.load_state_dict(torch.load(f"fga_combined_fold{fold}.pth", map_location=DEVICE))
            else:
                model = MetricLSTM().to(DEVICE)
                model.load_state_dict(torch.load(f"{metric_name}_fold{fold}.pth", map_location=DEVICE))
            
            model.eval()
            all_p, all_g = [], [] #pred and grad, easier to just run the model directly
            with torch.no_grad():
                for x, lengths, y in val_loader:
                    p = model(x.to(DEVICE), lengths)
                    all_p.extend(p.squeeze().cpu().numpy().reshape(-1))
                    all_g.extend(y.squeeze().cpu().numpy().reshape(-1))
            
            sample_pids = ds.df.iloc[val_idx]['participant_id'].values
            temp_df = pd.DataFrame({'pid': sample_pids, 'gt': all_g, 'pred': all_p}) 
            patient_results = temp_df.groupby('pid').mean().reset_index() #basically because multiple videos per patient, combine all preds into 1
            
            fold_gt[fold] = patient_results['gt'].values #this is just storing the final results per fold 
            fold_preds[fold] = patient_results['pred'].values
            fold_idx_map[fold] = patient_results['pid'].map(pid_to_csv_idx).values
            
        return fold_gt, fold_preds, fold_idx_map

def get_baseline_fold_predictions():
    ds = SplitManagerDataset(FEATURE_CSV, SPLIT_CSV, target_metric="fga_label")
    split_df = pd.read_csv(SPLIT_CSV)
    pid_to_csv_idx = {pid: idx for idx, pid in enumerate(split_df['participant_id'])}

    fold_gt = {}
    fold_preds = {}
    fold_idx_map = {}

    for fold in range(5):
        train_idx = np.where(ds.fold_assignments != fold)[0] #get train,val indices
        val_idx = np.where(ds.fold_assignments == fold)[0]

        train_mean = ds.labels[train_idx].mean() #get train mean or the baseline value

        sample_pids = ds.df.iloc[val_idx]['participant_id'].values
        val_gt_raw = ds.labels[val_idx] #get patient ground truth
        
        temp_df = pd.DataFrame({'pid': sample_pids, 'gt': val_gt_raw}) 
        patient_results = temp_df.groupby('pid').mean().reset_index() #put together all of the different patient values

        #all pred needs to be the train mean for baseline
        fold_gt[fold] = patient_results['gt'].values 
        fold_preds[fold] = np.full(shape=len(patient_results), fill_value=train_mean)
        fold_idx_map[fold] = patient_results['pid'].map(pid_to_csv_idx).values

    return fold_gt, fold_preds, fold_idx_map

if __name__ == "__main__":
    set_seed(RANDOM_STATE)
    
    results_table = []
    
    tasks = [
        "baseline_mae",
        "imbalance_label",
        "speed_label",
        "lateral_deviation_label",
        "gait_deviation_label",
        "device_label",
        "fga_label",
        "combined_bath_label",
        "combined_zeno_label",
        "fusion_fga_label"
    ]
    
    for metric in tasks:
        print(f"Evaluating {metric}")
        if metric == "baseline_mae": fold_gt, fold_preds, fold_idx_map = get_baseline_fold_predictions()
        else: fold_gt, fold_preds, fold_idx_map = get_fold_predictions(metric)
        
        res = apply_kerry_bootstraps(fold_gt, fold_preds, fold_idx_map, metric)
        results_table.append(res)
    
    print(f"{'Metric / Architecture':<25} | {'MAE (95% CI)':<25} | {'MSE (95% CI)':<25}")
    for r in results_table:
        mae_str = f"{r['mae']:.3f} ({r['mae_ci'][0]:.2f}-{r['mae_ci'][1]:.2f})"
        r2_str = f"{r['r2']:.3f} ({r['r2_ci'][0]:.2f}-{r['r2_ci'][1]:.2f})"
        print(f"{r['metric']:<25} | {mae_str:<22} | {r2_str:<22}")
