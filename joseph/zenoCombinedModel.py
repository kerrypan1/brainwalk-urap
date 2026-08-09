import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.preprocessing import StandardScaler
from multiple import set_seed, RANDOM_STATE, DEVICE
import torch
import torch.nn as nn
import torch.optim as optim

SPLIT_CSV = "[redacted].csv"
ZENO_DIR = "[redacted]"
TARGET_FEATURES = [
    "singlesupportratiolr", "walkratiocmstepsminmean", "stridewidthcmsd",
    "stridelengthcmcv", "stridetimeseccv", "meanegvimean", "stridewidthcmmean",
    "cadencestepsminmean", "absolutesteplengthcmmean", "singlesupportmean",
    "stridelengthcmmean", "velocitycmsecmean", "stridevelocitycmsecmean"
]
class ZenoHead(nn.Module): #for the combining of the zeno metrics
    def __init__(self, input_dim=13):
        super(ZenoHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(), #want to try and learn relations between metrics to get to the FGA score
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_neural_heads():
    set_seed(RANDOM_STATE)
    df = pd.read_csv(SPLIT_CSV)
    df['fold'] = df['split'].str.extract('(\d+)').astype(int)
    
    with open(os.path.join(ZENO_DIR, "feature_names.json"), "r") as f:
        all_names = json.load(f)
    indices = [all_names.index(f) for f in TARGET_FEATURES]

    X_list, y_list, fold_list = [], [], [] #zeno data
    for _, row in df.iterrows():
        z_path = os.path.join(ZENO_DIR, f"{row['participant_id']}.npy")
        if os.path.exists(z_path):
            data = np.nanmean(np.load(z_path), axis=0)
            X_list.append(data[indices])
            y_list.append(row['label']) 
            fold_list.append(row['fold'])
            
    X = np.array(X_list)
    y = np.array(y_list)
    folds = np.array(fold_list)

    all_r2_scores = []
    all_mae_scores = []
    all_zeno_predictions = []

    for fold in range(5):
        train_mask = folds != fold
        val_mask = folds == fold

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        #scale so is comparable between metrics of different units 
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        #convert to Tensors
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE).view(-1, 1)
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(DEVICE)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE).view(-1, 1)

        #init neural head
        model = ZenoHead(input_dim=len(TARGET_FEATURES)).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        #train head
        model.train()
        for epoch in range(200): #small dataset -> needs more epochs than LSTM
            optimizer.zero_grad()
            preds = model(X_train_t)
            loss = criterion(preds, y_train_t)
            loss.backward()
            optimizer.step()
        
        #eval
        model.eval()
        with torch.no_grad():
            y_pred_t = model(X_val_t)
            y_pred = y_pred_t.cpu().numpy().flatten()
            
            mae = np.mean(np.abs(y_pred - y_val))
            from sklearn.metrics import r2_score 
            fold_r2 = r2_score(y_val, y_pred) #see how much explained by each fold
            all_r2_scores.append(fold_r2)
            all_mae_scores.append(mae)
            print(f"Fold {fold} | MAE: {mae:.4f} | R2: {fold_r2:.4f}")

        joblib.dump(scaler, f"zeno_scaler_fold{fold}.pkl") #quick snapshot of just mean and std dev
        torch.save(model.state_dict(), f"zeno_head_fold{fold}.pth") #weights to save
        
        #store predictions for Fusion
        fold_pids = df.iloc[val_mask]['participant_id'].values
        for pid, p, a in zip(fold_pids, y_pred, y_val):
            all_zeno_predictions.append({
                "participant_id": pid,
                "zeno_pred": p,
                "actual_fga": a,
                "fold": fold
            })

    all_fold_importances = []
    for fold in range(5):
        model = ZenoHead(input_dim=len(TARGET_FEATURES))
        model.load_state_dict(torch.load(f"zeno_head_fold{fold}.pth"))
        
        w0 = model.net[0].weight.data #weights from inputs to hidden
        w2 = model.net[2].weight.data #weights from hidden to output
        
        directionality = (w0 * w2.t()).mean(dim=0).cpu().numpy() #if pos or neg -> at end can see magnitude
        all_fold_importances.append(directionality) #note that scaling is what makes the importance comparable, only inputs scaled so most we can say about that

    avg_importance = np.mean(all_fold_importances, axis=0)
    sorted_indices = np.argsort(avg_importance)[::-1] # High positive to high negative
    
    print(f"Average 5-Fold CV Combined MAE: {np.mean(all_mae_scores):.4f}")

    print("NEURAL FEATURE DIRECTIONALITY")
    print(f"{'Feature Name':<25} | {'Impact on FGA':<15}")
    for idx in sorted_indices:
        sign = "+" if avg_importance[idx] > 0 else "-"
        print(f"{TARGET_FEATURES[idx]:<25} | {avg_importance[idx]:>8.4f} ({sign})")
    
    zeno_preds_df = pd.DataFrame(all_zeno_predictions)
    zeno_preds_df.to_csv("zeno_predictions.csv", index=False)
    print("Saved 'zeno_predictions.csv'")

if __name__ == "__main__":
    train_neural_heads()
