import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

from fgaFinding import SplitManagerDataset, get_weighted_sampler, FEATURE_CSV, SPLIT_CSV, new_collate_fn
from multiple import DEVICE, set_seed, RANDOM_STATE
from combined import FGA_Estimator


def evaluate_fga_jury():
    set_seed(RANDOM_STATE)
    ds = SplitManagerDataset(FEATURE_CSV, SPLIT_CSV, target_metric="fga_label")
    
    video_oof_predictions = [] #out of fold pred
    fold_maes = []
    all_fold_weights = []
    all_fold_biases = []
    
    for fold in range(5): 
        #match weight filenames to fgaFinding format for consistency in cross val
        imb_weights = f"imbalance_label_fold{fold}.pth" 
        spd_weights = f"speed_label_fold{fold}.pth"
        lat_weights = f"lateral_deviation_label_fold{fold}.pth"
        gait_weights = f"gait_deviation_label_fold{fold}.pth"
        dev_weights = f"device_label_fold{fold}.pth"

        model = FGA_Estimator(imb_weights, spd_weights, lat_weights, gait_weights, dev_weights).to(DEVICE)
        optimizer = torch.optim.Adam(model.fga_head.parameters(), lr=1e-2)
        #mse for train as better learning vs mae as smoother + don't need to worry as much about interpretability here 
        criterion = nn.MSELoss()

        train_idx = np.where(ds.fold_assignments != fold)[0] 
        val_idx = np.where(ds.fold_assignments == fold)[0]

        train_loader = DataLoader(Subset(ds, train_idx), batch_size=16, 
                                  sampler=get_weighted_sampler(ds.labels[train_idx]), 
                                  collate_fn=new_collate_fn)
        val_loader = DataLoader(Subset(ds, val_idx), batch_size=16, shuffle=False, collate_fn=new_collate_fn)

        #train specifically fga_head while keeping base models frozen
        model.train()
        for epoch in range(20): #do more epochs here as only need to train the linear head so 10 weights real fast
            for x, lengths, y in train_loader:
                optimizer.zero_grad()
                pred = model(x.to(DEVICE), lengths)
                loss = criterion(pred.squeeze(), y.to(DEVICE).squeeze())
                loss.backward()
                optimizer.step()

        #store head weights and bias -> interpretability
        fold_weight = model.fga_head.weight.detach().cpu().numpy()[0]
        fold_bias = model.fga_head.bias.detach().cpu().numpy()[0]
        all_fold_weights.append(fold_weight)
        all_fold_biases.append(fold_bias)

        torch.save(model.state_dict(), f"fga_combined_fold{fold}.pth")

        #eval on mae not mse as we actually want mae to focus on
        model.eval()
        errors = []
        val_pids = ds.df.iloc[val_idx]['participant_id'].values
        fold_preds = [] 
        fold_gt = []
        
        with torch.no_grad():
            for x, lengths, y in val_loader:
                pred = model(x.to(DEVICE), lengths).squeeze()
                y_true = y.to(DEVICE).squeeze()
                
                batch_errors = torch.abs(pred - y_true).cpu().numpy() #calc all at once
                
                #standardize single vs multi batch 
                if batch_errors.ndim == 0: #single sample
                    errors.append(batch_errors.item())
                    fold_preds.append(pred.item())
                    fold_gt.append(y_true.item())
                else:
                    errors.extend(batch_errors) #multi sample
                    fold_preds.extend(pred.cpu().numpy())
                    fold_gt.extend(y_true.cpu().numpy())
        
        #'val' set so store for later 
        for pid, pred_val, actual_val in zip(val_pids, fold_preds, fold_gt):
            video_oof_predictions.append({
                "participant_id": pid,
                "video_pred": pred_val,
                "actual_fga": actual_val,
                "fold": fold
            })
            
        #still need to log and store the current fold results 
        mae = np.mean(errors)
        fold_maes.append(mae)
        print(f"Fold {fold} Combined MAE: {mae:.4f}")

    #report final cross val MAE with std dev
    print(f"\nAverage 5-Fold CV Combined MAE: {np.mean(fold_maes):.4f} (+/- {np.std(fold_maes):.4f})")
    
    #combined pred for later (ex fusion mdoel)
    video_preds_df = pd.DataFrame(video_oof_predictions)
    video_preds_df.to_csv("video_predictions.csv", index=False)

    #calc avg feature importance across folds
    avg_weights = np.mean(all_fold_weights, axis=0)
    avg_bias = np.mean(all_fold_biases)

    print("\n5-Fold Average: Feature Importance")
    print(f"1. Imbalance:         {avg_weights[0]:.4f}")
    print(f"2. Speed:             {avg_weights[1]:.4f}")
    print(f"3. Lateral Deviation: {avg_weights[2]:.4f}")
    print(f"4. Gait Deviation:    {avg_weights[3]:.4f}")
    print(f"5. Device:            {avg_weights[4]:.4f}") 
    print(f"Jury Baseline (Bias): {avg_bias:.4f}")

    features = ["Imbalance", "Speed", "Lateral Deviation", "Gait Deviation", "Device"]
    best_feature_idx = np.argmax(np.abs(avg_weights))
    print(f"Top Clinical Driver: {features[best_feature_idx]}")


if __name__ == "__main__":
    evaluate_fga_jury()