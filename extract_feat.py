# Import necessary libraries
from numpy.fft import rfft, rfftfreq
from sklearn import preprocessing
import numpy as np
import os
import pandas as pd
from pathlib import Path
import glob
from extract_1 import FFT

# Define the data path
data_path = Path('data/VBL-VA001')

# Initialize counters
total_files = 0
total_dirs = 0

# Walk through the directory
for base, dirs, files in os.walk(data_path):
    print(f'Searching in: {base}')
    total_dirs += len(dirs)
    total_files += len(files)

print(f'Total number of files: {total_files}')
print(f'Total number of directories: {total_dirs}')

# Collecting number data
def count_files_in_directory(directory):
    if directory.exists() and directory.is_dir():
        return len([entry for entry in directory.iterdir() if entry.is_file()])
    else:
        print(f"Directory {directory} does not exist.")
        return 0

dir_path1 = data_path / 'normal'
print(f'Total data Normal: {count_files_in_directory(dir_path1)}')

dir_path2 = data_path / 'misalignment'
print(f'Total data Misalignment: {count_files_in_directory(dir_path2)}')

dir_path3 = data_path / 'unbalance'
print(f'Total data Unbalance: {count_files_in_directory(dir_path3)}')

dir_path4 = data_path / 'bearing'
print(f'Total data Bearing: {count_files_in_directory(dir_path4)}')

# Load CSV files
normal_files = list(dir_path1.glob('*.csv')) if dir_path1.exists() else []
misalignment_files = list(dir_path2.glob('*.csv')) if dir_path2.exists() else []
unbalance_files = list(dir_path3.glob('*.csv')) if dir_path3.exists() else []
bearing_files = list(dir_path4.glob('*.csv')) if dir_path4.exists() else []

# Example function to process files
def load_and_process_files(file_list):
    data = pd.DataFrame()
    for file in file_list:
        df = pd.read_csv(file, header=None)
        data = pd.concat([data, df], axis=0, ignore_index=True)
    return data

# Process files
normal_data = load_and_process_files(normal_files)
misalignment_data = load_and_process_files(misalignment_files)
unbalance_data = load_and_process_files(unbalance_files)
bearing_data = load_and_process_files(bearing_files)

print(f'Normal data shape: {normal_data.shape}')
print(f'Misalignment data shape: {misalignment_data.shape}')
print(f'Unbalance data shape: {unbalance_data.shape}')
print(f'Bearing data shape: {bearing_data.shape}')

# Ensure variables are defined
normal = normal_data if not normal_data.empty else None
misalignment = misalignment_data if not misalignment_data.empty else None
unbalance = unbalance_data if not unbalance_data.empty else None
bearing = bearing_data if not bearing_data.empty else None

# Combine all conditions if they exist
all_cond = [cond for cond in [normal, misalignment, unbalance, bearing] if cond is not None]

if all_cond:
    combined_data = pd.concat(all_cond, axis=0, ignore_index=True)
    print(f'Combined data shape: {combined_data.shape}')
else:
    print('No data to combine.')


# Feature Extraction function
def std(data):
    '''Standard Deviation features'''
    data = np.asarray(data)
    stdev = pd.DataFrame(np.std(data, axis=1))
    return stdev


def mean(data):
    '''Mean features'''
    data = np.asarray(data)
    M = pd.DataFrame(np.mean(data, axis=1))
    return M


def pp(data):
    '''Peak-to-Peak features'''
    data = np.asarray(data)
    PP = pd.DataFrame(np.max(data, axis=1) - np.min(data, axis=1))
    return PP


def Variance(data):
    '''Variance features'''
    data = np.asarray(data)
    Var = pd.DataFrame(np.var(data, axis=1))
    return Var


def rms(data):
    '''RMS features'''
    data = np.asarray(data)
    Rms = pd.DataFrame(np.sqrt(np.mean(data**2, axis=1)))
    return Rms


def Shapef(data):
    '''Shape factor features'''
    data = np.asarray(data)
    shapef = pd.DataFrame(rms(data)/Ab_mean(data))
    return shapef


def Impulsef(data):
    '''Impulse factor features'''
    data = np.asarray(data)
    impulse = pd.DataFrame(np.max(data)/Ab_mean(data))
    return impulse


def crestf(data):
    '''Crest factor features'''
    data = np.asarray(data)
    crest = pd.DataFrame(np.max(data)/rms(data))
    return crest


def kurtosis(data):
    '''Kurtosis features'''
    data = pd.DataFrame(data)
    kurt = data.kurt(axis=1)
    return kurt


def skew(data):
    '''Skewness features'''
    data = pd.DataFrame(data)
    skw = data.skew(axis=1)
    return skw


# Helper functions to calculate features
def Ab_mean(data):
    data = np.asarray(data)
    Abm = pd.DataFrame(np.mean(np.absolute(data), axis=1))
    return Abm


def SQRT_AMPL(data):
    data = np.asarray(data)
    SQRTA = pd.DataFrame((np.mean(np.sqrt(np.absolute(data, axis=1))))**2)
    return SQRTA


def clearancef(data):
    data = np.asarray(data)
    clrf = pd.DataFrame(np.max(data, axis=1)/SQRT_AMPL(data))
    return clrf


# Extract features from X, Y, Z axis
def read_data(filenames):
    data = pd.DataFrame()
    for filename in filenames:
        df = pd.read_csv(filename, usecols=[1], header=None)
        data = pd.concat([data, df], axis=1, ignore_index=True)
    return data

# read data from csv files
all_cond = [normal, misalignment, unbalance, bearing]
cond_names = ['normal', 'misalignment', 'unbalance', 'bearing']
data = {}
fft = {}

for cond, cond_name in zip(all_cond, cond_names):
    for ax in ['x', 'y', 'z']:
        name = f"{cond_name}_{ax}"
        data[name] = read_data(cond).T.dropna(axis=1)
        fft[name] = FFT(data[name])

# fft_merged = pd.concat(fft, axis=1)

# Find max and min value of fft
max_value = max(fft.values(), key=lambda item: max(max(sub_array) for sub_array in item))
MAX_FFT = max(max(sub_array) for sub_array in max_value)
min_value = min(fft.values(), key=lambda item: min(min(sub_array) for sub_array in item))
MIN_FFT = min(min(sub_array) for sub_array in min_value)

def NormalizeData(**kwargs):  # Normalisasi (0-1)
    return (data - MIN_FFT) / (MAX_FFT - MIN_FFT)

