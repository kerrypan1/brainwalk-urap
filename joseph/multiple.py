import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random
import copy
import os

 #most of this is carry over from past usage

#config
LABELS_CSV = "[redacted].csv"
GAIT_INPUT_SIZE = 66
HIDDEN_SIZE = 8
NUM_LAYERS = 1
BATCH_SIZE = 16
EPOCHS = 60
LR = 0.002
PATIENCE = 15
RANDOM_STATE = 42
REPEAT = 5
DEVICE = torch.device("cpu") 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

class MetricDataset(Dataset):
    def __init__(self, csv_path, augment=False, target_metric="imbalance_label"):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=[target_metric]).reset_index(drop=True) #don't want errors as some of gait deviation not there
        
        self.augment = augment
        self.data_cache = [np.load(p) for p in self.df["pose_path"]]
        self.subjects = self.df["subject_id"].values if "subject_id" in self.df.columns else np.arange(len(self.df))

        self.labels = self.df[target_metric].values.astype(np.float32) #now is linear regression so needs to be scaled from 0.0-3.0 not classification

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.data_cache[idx].copy()
        y = self.labels[idx]

        if self.augment:
            stride = random.choice([2, 3, 4]) #temporal augmentation, make how many frames available different
            start = random.randint(0, min(stride - 1, len(x) - 1))
            x = x[start::stride]

            if random.random() < 0.3 and len(x) > 10:
                s = random.randint(0, len(x) - 10) #occlusion, cover some stuff up
                x[s : s + 10] = 0

            if random.random() < 0.5: # Horizontal Flip
                 x[..., 0] *= -1
            
            if random.random() < 0.5: #jitter
                x += np.random.normal(0, 0.01, x.shape)

            if random.random() < 0.5: #zoom
                x *= np.random.uniform(0.95, 1.05)

            if random.random() < 0.3: #temporal dropout or losing some frames (simulate glitch)
                mask = np.random.choice([0, 1], size=(x.shape[0], 1), p=[0.05, 0.95])
                x *= mask
        else:
            x = x[0::3] #standard for validation

        x = x.astype(np.float32)
        if x.std() > 0:
            x = (x - x.mean()) / (x.std() + 1e-6) #z score it so is easier to train and x,y scaled the same

        return torch.tensor(x), torch.tensor(y, dtype=torch.float32)

def collate_fn(batch): #turn samples from batch to a torch tensor
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs])
    xs = pad_sequence(xs, batch_first=True)
    ys = torch.stack(ys)
    return xs, lengths, ys

class MetricLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(GAIT_INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        out = self.dropout(h_n[-1]) #take last hidden state as output
        return self.head(out).squeeze(1) #squeeze to make sure that the output isn't 2d but 1d (or (16,1) vs 16)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    loss_sum = 0
    for x, l, y in loader:
        optimizer.zero_grad()
        logits = model(x.to(DEVICE), l)
        loss = criterion(logits, y.to(DEVICE))
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    return loss_sum / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    for x, l, y in loader:
        out = model(x.to(DEVICE), l)
        y_true.append(y.numpy())
        y_pred.append(out.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "Acc_0.5": np.mean(np.abs(y_true - y_pred) < 0.5)
    }

def main():
    set_seed(RANDOM_STATE)
    METRICS = [
        "imbalance_label", 
        "speed_label", 
        "gait_deviation_label",
        "lateral_deviation_label", 
        "device_label",
        "fga_label"
    ]

    for metric in METRICS:
        print(f"\n Training Regression for: {metric}")
        all_fold_results = []

        for repeat in range(REPEAT):
            ds_train = MetricDataset(LABELS_CSV, augment=True, target_metric=metric)
            ds_val = MetricDataset(LABELS_CSV, augment=False, target_metric=metric)

            #round the labels to integers just for the stratified split logic
            strat_labels = np.round(ds_train.labels).astype(int) 
            skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE + repeat)

            for fold, (tr, va) in enumerate(skf.split(ds_train.df, strat_labels, ds_train.subjects)):  
                train_loader = DataLoader(Subset(ds_train, tr), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(Subset(ds_val, va), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

                model = MetricLSTM().to(DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
                criterion = nn.MSELoss()

                best_mae = float('inf')
                wait = 0
                best_state = None

                for epoch in range(EPOCHS):
                    train_epoch(model, train_loader, optimizer, criterion)
                    res = evaluate(model, val_loader)

                    if res["MAE"] < best_mae:
                        best_mae = res["MAE"]
                        best_state = copy.deepcopy(model.state_dict())
                        wait = 0
                    else:
                        wait += 1
                    
                    if wait >= PATIENCE: 
                        break

                model.load_state_dict(best_state)
                torch.save(best_state, f"regression{metric}_fold{fold}_rep{repeat}.pth")
                all_fold_results.append(evaluate(model, val_loader))
                print(f"Repeat {repeat} Fold {fold} | Best MAE: {best_mae:.4f}")

        df_res = pd.DataFrame(all_fold_results)
        print(f"\n {metric.upper()} SUMMARY (15 Folds)")
        print(df_res.mean().round(4))
        print("+/- (Std)")
        print(df_res.std().round(4))

if __name__ == "__main__":
    main()
