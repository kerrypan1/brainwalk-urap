import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import PredefinedSplit, cross_val_predict
from multiple import set_seed, RANDOM_STATE

set_seed(RANDOM_STATE)

zeno_df = pd.read_csv("zeno_predictions.csv") #combining preds 
video_df = pd.read_csv("video_predictions.csv")

fusion_df = pd.merge(zeno_df, video_df, on=["participant_id", "fold", "actual_fga"]) #note as merged on kerry folds -> keeps folds

X = fusion_df[['zeno_pred', 'video_pred']].values
y = fusion_df['actual_fga'].values
folds = fusion_df['fold'].values

ps = PredefinedSplit(test_fold=folds)
meta_model = LinearRegression() #easiest and most interpretable way is just linear regression

fused_predictions = cross_val_predict(meta_model, X, y, cv=ps)

mae = mean_absolute_error(y, fused_predictions)
r2 = r2_score(y, fused_predictions)

print("Final Multimodal Fusion Results")
print(f"Fused MAE: {mae:.4f} FGA points")
print(f"Fused R2:  {r2:.4f}\n")

meta_model.fit(X, y)
print("Modality Importance")
print(f"Zeno Weight: {meta_model.coef_[0]:.4f}")
print(f"Bath Weight: {meta_model.coef_[1]:.4f}\n")

def bootstrap_final(gt, preds, n_iterations=1000):
    mae_dist = []
    n_samples = len(gt)
    set_seed(RANDOM_STATE) #can't hurt to set it again
    
    for _ in range(n_iterations):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        curr_mae = np.mean(np.abs(preds[idx] - gt[idx]))
        mae_dist.append(curr_mae)
        
    return np.percentile(mae_dist, 2.5), np.percentile(mae_dist, 97.5)

lower_ci, upper_ci = bootstrap_final(y, fused_predictions)
print(f"Bootstrap fusion MAE: {mae:.4f} [{lower_ci:.4f} - {upper_ci:.4f}]")

fusion_df['fusion_pred'] = fused_predictions
fusion_df.to_csv("fusion_predictions.csv", index=False)