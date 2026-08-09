import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, Dataset
import numpy as np
import pandas as pd
import re

from multiple import (
    MetricLSTM, set_seed,
    DEVICE, BATCH_SIZE, RANDOM_STATE
)

FEATURE_CSV = "[redacted].csv"
SPLIT_CSV = "[redacted].csv"
TARGET = "fga_label" 

class SplitManagerDataset(Dataset): #to account for the kerry splits, make dataset to do easily
    def __init__(self, feature_csv, split_csv, target_metric="fga_label"):
        self.target_metric = target_metric
        self.df_features = pd.read_csv(feature_csv)
        self.df_splits = pd.read_csv(split_csv)
        
        #convert "fold_0" -> 0
        self.df_splits['split_idx'] = self.df_splits['split'].apply(
            lambda x: int(re.search(r'\d+', str(x)).group()) if pd.notna(x) else -1
        )

        #merge clinical features with fold assignments
        self.df = self.df_features.merge(
            self.df_splits[['participant_id', 'split_idx']], 
            left_on='bw_id', 
            right_on='participant_id', 
            how='inner'
        )

        self.df = self.df.dropna(subset=[self.target_metric])

        self.labels = self.df[target_metric].values.astype(np.float32)
        self.pose_paths = self.df['pose_path'].values
        self.fold_assignments = self.df['split_idx'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.pose_paths[idx]
        try:
            data = np.load(path)
        except:
            data = np.zeros((200, 66)) #if no path just put 0s for all poses 

        #standardize to 200 as ideally long enought to see enough walking but not forget beginning
        max_len = 200
        if data.shape[0] > max_len:
            data = data[:max_len]
        else:
            pad = np.zeros((max_len - data.shape[0], 66))
            data = np.vstack([data, pad])

        return torch.FloatTensor(data), max_len, torch.FloatTensor([self.labels[idx]])

def new_collate_fn(batch): #as now not xs, lengths, ys but need (data,length, label)
    data, lengths, labels = zip(*batch) #group all together
    
    data = torch.stack(data) #make tensor
    
    lengths = torch.tensor(lengths)
    labels = torch.stack(labels) #stack as already [1] tensor
    
    return data, lengths, labels

def get_weighted_sampler(labels):
    clean_labels = np.nan_to_num(labels).astype(int)
    counts = np.bincount(clean_labels, minlength=4)
    weights = 1. / (counts + 1e-6) #don't want to div by 0 so 1e-6 shouldn't affect too much while still avoiding error
    sample_weights = np.array([weights[t] for t in clean_labels])
    return WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(sample_weights))

def run_training(target=TARGET):
    set_seed(RANDOM_STATE)
    
    ds_master = SplitManagerDataset(FEATURE_CSV, SPLIT_CSV, target)

    fold_maes = []
    
    for fold_val in range(5): #used to do hold out, now is cross val so all 5
        train_idx = np.where(ds_master.fold_assignments != fold_val)[0] 
        val_idx = np.where(ds_master.fold_assignments == fold_val)[0]

        sampler = get_weighted_sampler(ds_master.labels[train_idx]) #loader
        train_loader = DataLoader(Subset(ds_master, train_idx), batch_size=BATCH_SIZE, sampler=sampler, collate_fn=new_collate_fn)
        val_loader = DataLoader(Subset(ds_master, val_idx), batch_size=BATCH_SIZE, shuffle=False, collate_fn=new_collate_fn)

        model = MetricLSTM().to(DEVICE) #model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        best_mae = float('inf')
        associated_mse = 0.0
        
        for epoch in range(50):
            model.train()
            train_losses = []
            for x, lengths, y in train_loader:
                optimizer.zero_grad()
                pred = model(x.to(DEVICE), lengths)
                loss = criterion(pred.squeeze(), y.to(DEVICE).squeeze())
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            model.eval() #validation 
            all_preds = []
            all_gt = [] #ground truth

            with torch.no_grad():
                for x, lengths, y in val_loader:
                    p = model(x.to(DEVICE), lengths)
                    
                    all_preds.extend(p.squeeze().cpu().numpy().reshape(-1))
                    all_gt.extend(y.squeeze().cpu().numpy().reshape(-1))

            all_preds = np.array(all_preds)
            all_gt = np.array(all_gt)

            val_mae = np.mean(np.abs(all_preds - all_gt))
            val_mse = np.mean((all_preds - all_gt)**2) 

            if val_mae < best_mae:
                best_mae = val_mae
                associated_mse = val_mse
                torch.save(model.state_dict(), f"{target}_fold{fold_val}.pth") #only need best one
        
        fold_maes.append(best_mae)
        print(f"FOLD {fold_val} | Best MAE: {best_mae:.4f} | Final MSE: {associated_mse:.4f}")
    
    avg_mae = np.mean(fold_maes)
    std_mae = np.std(fold_maes)
    return avg_mae, std_mae

if __name__ == "__main__":
    metrics = [
        "imbalance_label", 
        "speed_label", 
        "lateral_deviation_label", 
        "gait_deviation_label",
        "device_label",
        "fga_label" #this is more for combined vs individual comparison
    ]
    summary_results = {}

    for metric in metrics:
        print(f"Training for {metric}")
        avg, std = run_training(metric)
        summary_results[metric] = (avg, std)

    print(f"{'Metric':<25} | {'Average MAE (+/- SD)':<20}")
    for metric, (avg, std) in summary_results.items():
        print(f"{metric:<25} | {avg:.4f} +/- {std:.4f}")
