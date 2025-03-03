import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path, header=None)

def extract_conditions(data):
    """Extract different conditions from the data."""
    conditions = {
        'normal': data.iloc[:1000, :],
        'misalignment': data.iloc[1000:2000, :],
        'unbalance': data.iloc[2000:3000, :],
        'bearing': data.iloc[3000:4000, :]
    }
    return conditions

def plot_feature(conditions, feature_index, feature_name):
    """Plot a specific feature for all conditions."""
    plt.figure(figsize=(10, 6))
    for condition, data in conditions.items():
        plt.plot(data.iloc[:, feature_index], label=condition)
    plt.title(f'{feature_name} for Different Conditions')
    plt.xlabel('Sample Index')
    plt.ylabel(feature_name)
    plt.legend()
    plt.show()

def main():
    # Define the file path
    file_path = Path(r"data\feature_VBL-VA001.csv")
    
    # Load data
    data = load_data(file_path)
    
    # Extract conditions
    conditions = extract_conditions(data)
    
    # Plot skewness for z-axis (assuming the last column is skewness for z-axis)
    plot_feature(conditions, feature_index=-1, feature_name='Skewness for Z-axis')

if __name__ == "__main__":
    main()