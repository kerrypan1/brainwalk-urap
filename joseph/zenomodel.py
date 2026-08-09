import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pandas as pd
import numpy as np
import os
import json
import random
from sklearn.preprocessing import StandardScaler

from multiple import DEVICE, set_seed, RANDOM_STATE

VIDEO_CSV = "[redacted].csv"
SPLIT_CSV = "[redacted].csv"
ZENO_DIR = "[redacted]"

TARGET_ZENO_FEATURES = [
    "singlesupportratiolr", "walkratiocmstepsminmean", "stridewidthcmsd",
    "stridelengthcmcv", "stridetimeseccv", "meanegvimean", "stridewidthcmmean",
    "cadencestepsminmean", "absolutesteplengthcmmean", "singlesupportmean",
    "stridelengthcmmean", "velocitycmsecmean", "stridevelocitycmsecmean"
]

class ZenoDataset(Dataset):
    def __init__(self, video_csv, zeno_dir, split_csv, augment=False):
        self.df_video = pd.read_csv(video_csv)
        self.df_split = pd.read_csv(split_csv)
        self.augment = augment
        
        self.df = self.df_video.merge( #to assign split
            self.df_split[['participant_id', 'split']], 
            left_on='bw_id', 
            right_on='participant_id'
        )
        self.df['fold'] = self.df['split'].str.extract('(\d+)').astype(int)

        with open(os.path.join(zeno_dir, "feature_names.json"), "r") as f:
            all_names = json.load(f)
        self.target_indices = [all_names.index(f) for f in TARGET_ZENO_FEATURES]

        y_raw = [] #scaling output zeno metrics so weird scaling doesn't affect accuracy
        self.valid_indices = []
        for i, row in self.df.iterrows():
            z_path = os.path.join(zeno_dir, f"{row['bw_id']}.npy")
            if os.path.exists(z_path):
                data = np.load(z_path)
                y_raw.append(np.nanmean(data, axis=0)[self.target_indices])
                self.valid_indices.append(i)
        
        self.df = self.df.iloc[self.valid_indices].reset_index(drop=True)
        self.y_scaler = StandardScaler()
        self.y_scaled = self.y_scaler.fit_transform(np.array(y_raw))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row['bw_id'] 
        fga_score = row['fga_label'] #technically not needed but include for completeness
        
        x = np.load(row['pose_path']).copy().astype(np.float32)
        y = self.y_scaled[idx].copy().astype(np.float32)

        if self.augment:
            stride = random.choice([2, 3, 4]) #time
            start = random.randint(0, min(stride - 1, len(x) - 1))
            x = x[start::stride]

            if random.random() < 0.5: #jitter
                x += np.random.normal(0, 0.005, x.shape)

            if random.random() < 0.5: #scale
                x *= np.random.uniform(0.98, 1.02)

            if random.random() < 0.3 and len(x) > 10: #occlusion
                s = random.randint(0, len(x) - 10)
                x[s : s + 10] = 0
            #no horizontal flip as will mess with the deviation
        else:
            x = x[0::3] #standard sampling for val -> matches the metricdataset

        if x.std() > 0: #normalize
            x = (x - x.mean()) / (x.std() + 1e-6)

        return torch.tensor(x), torch.tensor(y), pid

class ZenoLSTM(nn.Module):
    def __init__(self, input_size=66, hidden_size=64, num_layers=1, num_targets=13):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(hidden_size, num_targets)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.head(self.dropout(h_n[-1]))

def collate_fn(batch): #already used new_collate_fn so just go back to collate_fn as not using old one
    # batch is now [(x, y, pid), (x, y, pid), ...]
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    xs, ys, pids = zip(*batch) # Unpack the pids here
    
    lengths = torch.tensor([len(x) for x in xs])
    xs_padded = pad_sequence(xs, batch_first=True)
    
    return xs_padded, lengths, torch.stack(ys), list(pids)


def run_training():
    set_seed(RANDOM_STATE)
    DEVICE = torch.device("cpu")
    
    train_dataset = ZenoDataset(VIDEO_CSV, ZENO_DIR, SPLIT_CSV, augment=True)
    val_dataset = ZenoDataset(VIDEO_CSV, ZENO_DIR, SPLIT_CSV, augment=False)
    
    cross_val_scores = []
    
    for fold in range(5):
        train_idx = train_dataset.df[train_dataset.df['fold'] != fold].index
        val_idx = val_dataset.df[val_dataset.df['fold'] == fold].index
        
        train_loader = DataLoader(
            Subset(train_dataset, train_idx), 
            batch_size=16, 
            shuffle=True, 
            pin_memory=False,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            Subset(val_dataset, val_idx), 
            batch_size=16, 
            shuffle=False, 
            pin_memory=False,
            collate_fn=collate_fn
        )

        model = ZenoLSTM().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss() #as training not eval
        
        best_mae = float('inf')
        
        for epoch in range(50):
            model.train()
            for x, l, y, pids in train_loader:
                optimizer.zero_grad()
                preds = model(x.to(DEVICE), l)
                loss = criterion(preds, y.to(DEVICE))
                loss.backward()
                optimizer.step()
            
            #validation
            model.eval()
            total_ae = 0
            with torch.no_grad():
                for x, l, y, pids in val_loader:
                    p = model(x.to(DEVICE), l)
                    p_real = val_dataset.y_scaler.inverse_transform(p.cpu().numpy()) #if don't compare back then no units to compare against
                    y_real = val_dataset.y_scaler.inverse_transform(y.cpu().numpy()) 
                    total_ae += np.sum(np.abs(p_real - y_real))
            
            avg_mae = total_ae / (len(val_idx) * 13) 
            
            if avg_mae < best_mae:
                best_mae = avg_mae
                torch.save(model.state_dict(), f"zeno_fold{fold}.pth")

        model.load_state_dict(torch.load(f"zeno_fold{fold}.pth"))
        model.eval()
        
        fold_results = []
        with torch.no_grad():
            for x, l, y, pids in val_loader:
                preds = model(x.to(DEVICE), l)
                
                p_real = val_dataset.y_scaler.inverse_transform(preds.cpu().numpy())
                y_real = val_dataset.y_scaler.inverse_transform(y.cpu().numpy())
                
                for i in range(len(pids)):
                    fold_results.append({
                        "participant_id": pids[i],
                        "zeno_pred": np.mean(p_real[i]), 
                        "fold": fold
                    })

        cross_val_scores.append(best_mae)
        print(f"Fold {fold}: Best Validation MAE: {best_mae:.4f}")

    print("CROSS-VALIDATION RESULTS (5 Folds)")
    print(f"Individual Folds: {[round(score, 4) for score in cross_val_scores]}")
    print(f"Overall CV MAE:   {np.mean(cross_val_scores):.4f} +/- {np.std(cross_val_scores):.4f}")

if __name__ == "__main__":
    run_training()
