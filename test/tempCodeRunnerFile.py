from sklearn import impute
from sklearn.base import is_classifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import wfdb
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from scipy.signal import butter, lfilter, filtfilt

# import model
import sys
sys.path.insert(0, "")

from test.models.gradient_boost import gradient_boosting_rhythm_classification
from test.synthetic_test_model.gradient_boost import evaluate_synthetic_test_set




# Function to normalize ECG signals
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

def extract_heart_rate(peaks, sampling_frequency):
    
    
    # Calculate R-R intervals (differences between consecutive R peaks)
    rr_intervals = np.diff(peaks) / sampling_frequency
    
    # Calculate mean heart rate
    if len(rr_intervals) > 0:
        mean_hr = 60 / np.mean(rr_intervals)
    else:
        mean_hr = 0  # Placeholder, in case no peaks are detected
    
    return mean_hr

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


# def classify_rhythm(rr_intervals, heart_rate):
#     if not rr_intervals.size:
#         return "No Data"
    
#     # Calculate standard deviation of RR intervals for arrhythmia detection
#     std_rr = np.std(rr_intervals)
    
#     # Define thresholds for heart rate (in beats per minute)
#     bradycardia_threshold = 60  # Heart rate below 60 bpm is considered bradycardia
#     tachycardia_threshold = 100  # Heart rate above 100 bpm is considered tachycardia
    
#     # Rhythm classification based on heart rate and RR interval variability
#     if heart_rate < bradycardia_threshold:
#         rhythm_category = "Bradycardia"
#     elif heart_rate > tachycardia_threshold:
#         rhythm_category = "Tachycardia"
#     else:
#         rhythm_category = "Normal Heart Rate"
    
#     # Further classify based on RR interval variability
#     if std_rr < 0.1:
#         rhythm_type = "Sinus Rhythm"
#     else:
#         rhythm_type = "Irregular Rhythm"  # Potential indicator of arrhythmias like atrial fibrillation
    
#     # Combine both classifications for a detailed description
#     combined_rhythm_classification = f"{rhythm_category}, {rhythm_type}"
    
#     return combined_rhythm_classification
def standard_deviation_rr_interval(rr_intervals):
    # Calculate the standard deviation of R-R intervals as a scalar
    std_rr = np.std(rr_intervals) if np.size(rr_intervals) > 1 else 0
    return std_rr

def classify_rhythm(heart_rate, std_rr):
    # Ensure heart_rate is a scalar
    if not np.isscalar(heart_rate):
        raise ValueError("heart_rate must be a scalar value.")
    
    # Criteria for normal rhythm classification
    normal_hr_min = 60
    normal_hr_max = 100
    
    # Criteria for Atrial Flutter (simplified for this example)
    afl_hr_min = 100
    afl_hr_max = 175
    afl_std_rr_max = 0.1  # Assuming a relatively regular R-R interval in Atrial Flutter
    
    if normal_hr_min <= heart_rate <= normal_hr_max and std_rr < afl_std_rr_max:
        return 'Normal'
    elif afl_hr_min <= heart_rate <= afl_hr_max and std_rr < afl_std_rr_max:
        return 'Atrial Flutter'
    else:
        return 'Other'

# Example use:
# Assuming you have the heart rate and R-R intervals for an individual record
# heart_rate_example = 75  # Example scalar heart rate
# rr_intervals_example = np.array([0.8, 0.81, 0.79, 0.82])  # Example R-R intervals array

# rhythm_classification = classify_rhythm(heart_rate_example, rr_intervals_example)
# print(f'Rhythm Classification: {rhythm_classification}')

    

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

# def find_p_waves(ecg_signal, r_peaks, fs):
#     p_wave_indices = []
#     search_window = int(0.2 * fs)  # Search window for P wave 200 ms before R peak

#     for r_peak in r_peaks:
#         search_region_start = max(0, r_peak - search_window)
#         search_region = ecg_signal[search_region_start:r_peak]
#         if len(search_region) == 0:
#             continue
#         p_peak = np.argmax(search_region) + search_region_start
#         p_wave_indices.append(p_peak)

#     return p_wave_indices

# def infer_anatomic_origin_with_p_wave(qrs_widths_ms, combined_rhythm_classification, p_wave_present):
#     origin = "Unknown"
    
#     # Extract rhythm category and type from the combined classification
#     rhythm_category, rhythm_type = combined_rhythm_classification.split(', ')
#     # Adjust logic based on detailed rhythm classification
#     if "Irregular Rhythm" in rhythm_type:
#         if qrs_widths_ms < 120 and p_wave_present:
#             origin = "Atrial - Possible Atrial Fibrillation with observable atrial activity"
#         elif not p_wave_present:
#             origin = "Atrial - Atrial Fibrillation (No observable P wave)"
#     elif qrs_widths_ms > 120:
#         if not p_wave_present:
#             origin = "Ventricular - Potential Ventricular Tachycardia (No P wave)"
#         else:
#             origin = "Abnormal - P wave present but with wide QRS, needs further investigation"
#     elif "Sinus Rhythm" in rhythm_type:
#         if qrs_widths_ms < 120 and p_wave_present:
#             origin = "Supraventricular - Normal Sinus Rhythm"
#         elif not p_wave_present:
#             origin = "Junctional or Ventricular - Abnormal (No P wave)"

#     return origin




# Directory where your ECG files are stored
ecg_directory = "small-data/records100/00000"
population_data_path = "small-data/data.csv"


population_df = preprocess_population_data(population_data_path)

# Modify the DataFrame to include a new column for linking with ECG data
population_df['record_name'] = population_df['filename_lr'].apply(extract_record_name_from_path)

ecg_features_list = []

# Iterate through the population_df to find matching ECG files
for index, row in population_df.iterrows():
    record_name = row['record_name']  # This is the linking key
    record_path = os.path.join(ecg_directory, record_name)
    # Only proceed if the file exists
    if os.path.exists(record_path + ".hea"):
        record = wfdb.rdrecord(record_path)
        ecg_signal = record.p_signal[:, 0]  # Assuming we're interested in the first lead
        sampling_frequency = record.fs
        
        # Normalize and process the ECG signal
        ecg_signal_normalized = normalize_signals(ecg_signal)

        filtered_signal = preprocess_signal_for_p_waves(ecg_signal_normalized, sampling_frequency)

        # find the peaks
        peaks, _ = find_peaks(filtered_signal, height=np.max(filtered_signal)/6, distance=sampling_frequency/3)
        # Calculate the heart rate
        heart_rate = extract_heart_rate(peaks, sampling_frequency)
        
        # Detect QRS complexes (Q and S points)
        # q_points, s_points = find_qrs_complexes(filtered_signal, peaks, sampling_frequency)
        # Calculate QRS widths in milliseconds
        # qrs_widths_ms = (s_points - q_points) /sampling_frequency * 1000
        # mean_qrs_width = np.mean(qrs_widths_ms)
        
        # R-R intervals for rhythm classification
        rr_intervals = np.diff(peaks) / sampling_frequency
        
        standard_deviation = standard_deviation_rr_interval(rr_intervals)
        # find rhythm based on heart rate and rr interval
        rhythm = classify_rhythm(heart_rate, standard_deviation)

        # find P-wave
        # P_wave = find_p_waves(filtered_signal, peaks, sampling_frequency)

        # Check if P-waves are present
        # p_wave_present = len(P_wave) > 0
        # anatomic_location = infer_anatomic_origin_with_p_wave(mean_qrs_width, rhythm, p_wave_present )

        # Append the extracted features to the list
        ecg_features_list.append({
            'record_name': record_name,
            'heart_rate': heart_rate,
            # 'qrs_width': mean_qrs_width,
            "standard_deviation": standard_deviation,
            'calculated_rhythm': rhythm,
            # 'anatomic_location': anatomic_location
        })

# Convert the list of dictionaries to a DataFrame
ecg_features_df = pd.DataFrame(ecg_features_list)

# Merge the ECG features with the population data on the 'record_name' column
combined_df = pd.merge(population_df, ecg_features_df, on='record_name', how='left')

# Print the combined DataFrame to verify
# print(combined_df.head(10))



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Re-define the rhythm mapping function for clarity and direct application to the new dataset
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

# Apply the rhythm mapping transformation to the scp_codes column in the new dataset
combined_df['rhythm_classification'] = combined_df['scp_codes'].apply(map_scp_to_rhythm)

# Display the first few rows of the new rhythm_classification column to verify the transformation
# print(combined_df[['scp_codes', 'rhythm_classification']].head(30))

# Filtering for records with 'Normal' or 'Atrial Flutter' rhythm classifications
filtered_df = combined_df[combined_df['rhythm_classification'].isin(['Normal', 'Atrial Flutter'])]

# Displaying the filtered DataFrame
print(filtered_df.head(10))
filtered_df.to_csv("small-data/filter_data.csv", index= False)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
 
# # Gradient Boosting for Rhythm Classification

predictions_df, xgb_classifier = gradient_boosting_rhythm_classification(filtered_df)

print(predictions_df[predictions_df['Actual Rhythm'].isin(['Atrial Flutter'])])
# Display some of the prediction results
# print(predictions_df.head(50))

#########################################################################################################################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Create a Synthetic Test Set:
# Define a function to generate heart rates based on rhythm
def generate_heart_rate(rhythm):
    if "Normal" in rhythm:
        return np.random.uniform(60, 100)  # Example: Random heart rate for bradycardia
    elif "Atrial Flutter" in rhythm:
        return np.random.uniform(100, 175)  # Example: Random heart rate for tachycardia
    else:
        return np.random.uniform(40, 60)  # Example: Random heart rate for normal rhythm

# Identify unique patterns from the provided DataFrame
unique_patterns = filtered_df[['rhythm_classification']].drop_duplicates()
print(unique_patterns)

# # # Generate synthetic data for each unique pattern
synthetic_data = []
for _, pattern in unique_patterns.iterrows():
    # Generate synthetic data points based on the pattern
    num_samples = 1000  # Number of synthetic samples per pattern
    rhythm = str(pattern['rhythm_classification'])  # Convert rhythm to string
    synthetic_samples = {
        'rhythm': np.repeat(rhythm, num_samples),
        # 'anatomic_location': np.repeat(pattern['anatomic_location'], num_samples),
        'age': np.random.randint(20, 80, num_samples),  # Example: Random age between 20 and 80
        'sex': np.random.choice([0, 1], num_samples),  # 0 for Male, 1 for Female
        'heart_rate': [generate_heart_rate(rhythm) for _ in range(num_samples)],  # Generate heart rates based on rhythm
        'weight': np.random.randint(40, 150, num_samples),
        'standard_deviation': np.random.uniform(0.01, 0.2, num_samples)
        # 'qrs_width': np.random.uniform(70, 120, num_samples),  # Example: Random QRS width between 70 and 120
    }
    synthetic_data.append(pd.DataFrame(synthetic_samples))

# Combine synthetic data for all patterns into a single DataFrame
synthetic_test_set = pd.concat(synthetic_data, ignore_index=True)

# Optionally, shuffle the synthetic test set if desired
synthetic_test_set = synthetic_test_set.sample(frac=1).reset_index(drop=True)

# Display synthetic test set
print("\nsynthetic_test_set: \n")
print(synthetic_test_set.head(5))


##################################################################################################
# validate test set with Gradient Boosting
comparison_df = evaluate_synthetic_test_set(synthetic_test_set, xgb_classifier, label_encoder)
print(comparison_df.head(10))

# #####################################################################################################################

