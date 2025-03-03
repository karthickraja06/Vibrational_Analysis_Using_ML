import argparse
import pandas as pd
import numpy as np
from numpy.fft import rfft
from scipy.stats import kurtosis, skew

def FFT(data):
    data = np.asarray(data)
    n = len(data)
    dt = 1 / 20000  # time increment in each data
    data = rfft(data) * dt
    return np.abs(data)

def extract_features(data):
    features = {
        'mean': np.mean(data, axis=1),
        'std': np.std(data, axis=1),
        'peak_to_peak': np.ptp(data, axis=1),
        'variance': np.var(data, axis=1),
        'rms': np.sqrt(np.mean(data**2, axis=1)),
        'abs_mean': np.mean(np.abs(data), axis=1),
        'shape_factor': np.sqrt(np.mean(data**2, axis=1)) / np.mean(np.abs(data), axis=1),
        'impulse_factor': np.max(data, axis=1) / np.mean(np.abs(data), axis=1),
        'crest_factor': np.max(data, axis=1) / np.sqrt(np.mean(data**2, axis=1)),
        'sqrt_amplitude': (np.mean(np.sqrt(np.abs(data)), axis=1))**2,
        'clearance_factor': np.max(data, axis=1) / (np.mean(np.sqrt(np.abs(data)), axis=1))**2,
        'kurtosis': kurtosis(data, axis=1),
        'skewness': skew(data, axis=1)
    }
    return pd.DataFrame(features)

def process_axis(data_csv, axis_indices):
    axis_data = data_csv.drop(data_csv.columns[axis_indices], axis=1).T
    axis_data = axis_data.dropna(axis=1)
    print(f"Processed axis data shape: {axis_data.shape}")
    return axis_data

def main(input_file):
    data_csv = pd.read_csv(input_file, header=None)
    print(f"Original data shape: {data_csv.shape}")

    # Process each axis
    test_x = process_axis(data_csv, [0, 2, 3])
    test_y = process_axis(data_csv, [0, 1, 3])
    test_z = process_axis(data_csv, [0, 1, 2])

    # Apply FFT
    fft_test_x = FFT(test_x)
    fft_test_y = FFT(test_y)
    fft_test_z = FFT(test_z)

    # Extract features
    features_x = extract_features(fft_test_x)
    features_y = extract_features(fft_test_y)
    features_z = extract_features(fft_test_z)

    # Combine features from all axes
    combined_features = pd.concat([features_x, features_y, features_z], axis=1, ignore_index=True)
    print(f"Combined features shape: {combined_features.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input CSV file")
    args = parser.parse_args()
    main(args.input)
