# Function to normalize ECG signals
import os
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from scipy.signal import butter, lfilter, filtfilt


def normalize_signals(ecg_signals):
    # Check if the input is a 1D array (a single lead)
    if ecg_signals.ndim == 1:
        # Normalize a single lead
        ecg_signals = (ecg_signals - np.mean(ecg_signals)) / np.std(ecg_signals)
    else:
        # Assuming ecg_signals is a 2D numpy array: rows are leads and columns are time points
        ecg_signals = (ecg_signals - np.mean(ecg_signals, axis=1, keepdims=True)) / np.std(ecg_signals, axis=1, keepdims=True)
    return ecg_signals


# Placeholder function for handling missing values
def handle_missing_values(ecg_signals):
    # Example: Replace missing values (if any) with the mean of the signal
    # This is just a placeholder. Actual implementation will depend on how missing values are represented
    if np.isnan(ecg_signals).any():
        for i in range(ecg_signals.shape[0]):
            if np.isnan(ecg_signals[i]).any():
                ecg_signals[i][np.isnan(ecg_signals[i])] = np.nanmean(ecg_signals[i])
    return ecg_signals


def preprocess_population_data(csv_path):
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Drop columns with a high percentage of missing values
    columns_to_drop = ['height', 'infarction_stadium2', 'validated_by', 'burst_noise', 'electrodes_problems', 'extra_beats', 'pacemaker']
    df = df.drop(columns=columns_to_drop)
    
    # Fill missing values for categorical data with 'Unknown'
    categorical_columns = ['heart_axis', 'baseline_drift']  # Add other categorical columns as needed
    df[categorical_columns] = df[categorical_columns].fillna('Unknown')
    
    # Standardize numerical features (age and weight)
    numerical_columns = ['age', 'weight']
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    return df

# New function to extract record name from filename columns
def extract_record_name_from_path(path):
    # Extracts the last part of the path and returns it (e.g., "00001_lr")
    return os.path.basename(path)


# def find_qrs_complexes(ecg_signal, r_peaks, fs):
#     q_points = []
#     s_points = []
#     qrs_width = int(0.1 * fs)  # 100 ms around R peak, adjust based on data
    
#     for r_peak in r_peaks:
#         # Define search regions for Q and S points around R peak
#         q_search_region = ecg_signal[max(0, r_peak - qrs_width):r_peak]
#         s_search_region = ecg_signal[r_peak:min(len(ecg_signal), r_peak + qrs_width)]
        
#         # Find Q point as minimum before R peak
#         if len(q_search_region) > 0:
#             q_point = np.argmin(q_search_region) + max(0, r_peak - qrs_width)
#             q_points.append(q_point)
        
#         # Find S point as minimum after R peak
#         if len(s_search_region) > 0:
#             s_point = np.argmin(s_search_region) + r_peak
#             s_points.append(s_point)
    
#     return np.array(q_points), np.array(s_points)





def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def preprocess_signal_for_p_waves(ecg_signal, fs):
    # Bandpass filter settings for P wave detection (frequencies between 1 Hz and 40 Hz are typically useful)
    lowcut = 1.0
    highcut = 40.0
    filtered_signal = butter_bandpass_filter(ecg_signal, lowcut, highcut, fs, order=2)
    return filtered_signal


def map_scp_to_rhythm(scp_codes_str):
    # Evaluate the string representation of the dictionary to a dictionary
    scp_codes = eval(scp_codes_str)
    
    # Define a mapping of SCP codes to rhythm classes
    rhythm_mapping = {
        'NORM': 'Normal',
        'AFIB': 'Atrial Fibrillation',
        'AFLT': 'Atrial Flutter',
        'SBRAD': 'Bradycardia',
        'STACH': 'Tachycardia',
        # Add more mappings as needed
    }
    
    # Default rhythm classification
    rhythm_classification = 'Other'
    
    # Check each SCP code in the record against the rhythm mapping
    for scp_code in scp_codes.keys():
        if scp_code in rhythm_mapping:
            rhythm_classification = rhythm_mapping[scp_code]
            break  # Stop at the first relevant rhythm classification found
    
    return rhythm_classification