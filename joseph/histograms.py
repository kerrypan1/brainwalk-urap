import pandas as pd
import matplotlib.pyplot as plt
import os

from zenomodel import SPLIT_CSV

ZENO_EXCEL = "[redacted].xlsx"
ZENO_SMALLER = "[redacted].csv"
TRAIN_CSV = "[redacted].csv"

TARGET_FEATURES = [
    "singlesupportratiolr", "walkratiocmstepsminmean", "stridewidthcmsd",
    "stridelengthcmcv", "stridetimeseccv", "meanegvimean", "stridewidthcmmean",
    "cadencestepsminmean", "absolutesteplengthcmmean", "singlesupportmean",
    "stridelengthcmmean", "velocitycmsecmean", "stridevelocitycmsecmean"
]
BATH_METRICS = [
    "imbalance_label", 
    "speed_label", 
    "lateral_deviation_label", 
    "gait_deviation_label",
    "device_label",
    "fga_label"
]


def get_zeno_data(): #as annoying to read every time
    if os.path.exists(ZENO_SMALLER):
        return pd.read_csv(ZENO_SMALLER)
    else:
        df = pd.read_excel(ZENO_EXCEL)
        subset = df[['bw_id'] + [f for f in TARGET_FEATURES if f in df.columns]]
        subset.to_csv(ZENO_SMALLER, index=False)
        return subset

def plot_zeno_histograms():
    zeno_df = get_zeno_data()
    for feature in TARGET_FEATURES:
        if feature in zeno_df.columns: #basically just make a hist for each feature 
            plt.figure(figsize=(8, 4))
            zeno_df[feature].hist(bins=30, color='skyblue', edgecolor='black')
            plt.title(f"Distribution of {feature}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.savefig(f"hist_{feature}.png")
            plt.close()

def plot_zeno_combined_raw_histogram(): #as the final MAE for the combined was mixed so need to match that 
    zeno_df = get_zeno_data()
    
    all_raw_values = []
    
    for feature in TARGET_FEATURES:
        if feature in zeno_df.columns: #just get all non NaNs and put in 1 list
            values = zeno_df[feature].dropna().tolist()
            all_raw_values.extend(values)
    
    if all_raw_values:
        plt.figure(figsize=(10, 6))
        plt.hist(all_raw_values, bins=100, color='gray', edgecolor='black', alpha=0.7)
        
        plt.title("Combined Raw Distribution of All 13 Zeno Metrics")
        plt.xlabel("Raw Metric Value (Mixed Units: cm, sec, steps/min)") #important to note that values aren't unified units 
        plt.ylabel("Frequency (Count across all metrics)")

        plt.savefig("hist_zeno_combined_raw.png")
        plt.close()

def plot_bath_histograms():
    df = pd.read_csv(TRAIN_CSV)

    for metric in BATH_METRICS:
        if metric in df.columns:
            counts = df[metric].value_counts().sort_index() #for bar chart as only have 3-4 labels 
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(counts.index.astype(str), counts.values, color='salmon', edgecolor='black') #need index str for good spacing
            
            ax.set_title(f"Distribution: {metric}")
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            
            plt.tight_layout() #saving
            plt.savefig(f"hist_{metric}.png")
            plt.close(fig)

if __name__ == "__main__":
    plot_zeno_histograms() 
    plot_bath_histograms()
    plot_zeno_combined_raw_histogram()
