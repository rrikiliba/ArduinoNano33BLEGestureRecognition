import pandas as pd
import numpy as np
from scipy.signal import welch
import os
import glob

WINDOW_SIZE = 60 # 60 samples window size (~0.5 seconds considering the IMU rate of 119hz)
STEP_SIZE = 30   # 50% overlap

DATA_DIR = 'dataset'
OUTPUT_FILE = 'features.csv' 

def compute_features(window):
    """ Computes Mean, Std, RMS, Min, Max, and PSD peak for a 1D array."""
    mean_val = np.mean(window)
    std_val = np.std(window)
    rms_val = np.sqrt(np.mean(window**2))
    min_val = np.min(window)
    max_val = np.max(window)
    _, psd_values = welch(window, fs=119, nperseg=len(window))
    psd_peak = np.max(psd_values)

    return [mean_val, std_val, rms_val, min_val, max_val, psd_peak]

def process_file(filepath, label):
    """ Applies sliding window to the CSV file and extracts features per window."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Skipping.")
        return []

    print(f"Processing {filepath}...")
    # Explicitly read the label from the filename and handle backslashes for Windows paths
    filename = os.path.basename(filepath)
    df = pd.read_csv(filepath)

    cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']
    data = df[cols].values

    features_list = []

    # sliding window
    for i in range(0, len(data) - WINDOW_SIZE + 1, STEP_SIZE):
        window = data[i:i + WINDOW_SIZE]

        window_features = []

        for axis_idx in range(6):
            axis_data = window[:, axis_idx]
            axis_features = compute_features(axis_data)
            window_features.extend(axis_features)

        # add correlation between X and Y axes, useful for detecting circles
        corr_aXY = np.corrcoef(window[:, 0], window[:, 1])[0, 1]
        if np.isnan(corr_aXY): corr_aXY = 0.0
        corr_gXY = np.corrcoef(window[:, 3], window[:, 4])[0, 1]
        if np.isnan(corr_gXY): corr_gXY = 0.0

        window_features.extend([corr_aXY, corr_gXY])
        window_features.append(label)
        features_list.append(window_features)

    return features_list

def main():
    all_features = []

    for filepath in glob.glob(f'{DATA_DIR}/*'):
        filename = os.path.basename(filepath)
        label = filename.split('.')[0]
        extracted = process_file(filepath, label)
        all_features.extend(extracted)

    if not all_features:
        print("No data processed. Please check your CSV files.")
        return

    axes = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']
    feature_names = ['mean', 'std', 'rms', 'min', 'max', 'psd']

    col_names = []
    for axis in axes:
        for fname in feature_names:
            col_names.append(f"{axis}_{fname}")            
    col_names.extend(['corr_aXY', 'corr_gXY', 'label'])

    result_df = pd.DataFrame(all_features, columns=col_names)
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFeature extraction complete. Extracted {len(result_df)} windows.")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()