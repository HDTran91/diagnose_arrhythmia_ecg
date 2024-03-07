from sklearn import impute
from sklearn.base import is_classifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import wfdb
from wfdb import processing
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
import heartpy as hp
from biosppy.signals import ecg
import neurokit2 as nk
import nolds



# import model
import sys
sys.path.insert(0, "")

from test.models.gradient_boost import gradient_boosting_rhythm_classification
from test.synthetic_test_model.gradient_boost import evaluate_synthetic_test_set
from test.synthetic_test_model.synthetic_test_generator import generate_synthetic_data
from test.functions.preprocessing import map_scp_to_rhythm, normalize_signals
from test.functions.preprocessing import preprocess_population_data
from test.functions.preprocessing import extract_record_name_from_path
from test.functions.preprocessing import preprocess_signal_for_p_waves
from test.functions.calculated import extract_atrial_fibrillation_features, extract_atrial_flutter_features, extract_normal_rhythm_features, extract_sinus_tachycardia_features


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

        # Process the ECG signal
        out = ecg.ecg(signal=filtered_signal, sampling_rate=sampling_frequency, show=False)

        # extract_normal_rhythm_features
        heart_rate, p_wave_amplitude, p_wave_duration, qrs_duration, pr_interval, qt_c = extract_normal_rhythm_features(out, sampling_frequency)
        
        # Extracted features for Atrial Flutter
        f_wave_frequency, atrial_rate = extract_atrial_flutter_features(out)

        # Extracted features for Atrial Fibrillation
        irregular_rhythm, p_wave_presence, rr_interval_variability = extract_atrial_fibrillation_features(out, sampling_frequency)

        # Extracted features for Sinus Tachycardia
        increased_heart_rate, p_wave_normal_morphology = extract_sinus_tachycardia_features(out)

        ecg_features_list.append({
            "record_name": record_name,
            "heart_rate": heart_rate,
            "p_wave_amplitude": p_wave_amplitude,
            "p_wave_duration": p_wave_duration,
            "qrs_duration": qrs_duration,
            "pr_interval": pr_interval,
            "qt_c": qt_c,
            "f_wave_frequency": f_wave_frequency,
            "atrial_rate": atrial_rate,
            "irregular_rhythm": irregular_rhythm,
            "p_wave_presence": p_wave_presence,
            "rr_interval_variability": rr_interval_variability,
            "increased_heart_rate": increased_heart_rate,
            "p_wave_normal_morphology": p_wave_normal_morphology
        })

# Convert the list of dictionaries to a DataFrame
ecg_features_df = pd.DataFrame(ecg_features_list)
print(ecg_features_df.head(50))

# Merge the ECG features with the population data on the 'record_name' column
combined_df = pd.merge(population_df, ecg_features_df, on='record_name', how='left')

# Apply the rhythm mapping transformation to the scp_codes column in the new dataset
combined_df['rhythm_classification'] = combined_df['scp_codes'].apply(map_scp_to_rhythm)

# Filtering for records with 'Normal' or 'Atrial Flutter' rhythm classifications
filtered_df = combined_df[combined_df['rhythm_classification'].isin(['Normal','Atrial Flutter', 'Atrial Fibrillation', 'Sinus Tachycardia'])]



# print(filtered_df[filtered_df['rhythm_classification'].isin(['Sinus Tachycardia'])].head(50))
print(filtered_df.head(50))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
 
# # Gradient Boosting for Rhythm Classification

predictions_df, xgb_classifier, label_encoder = gradient_boosting_rhythm_classification(filtered_df)

# # Display some of the prediction results
# # print(predictions_df[predictions_df['Actual Rhythm'].isin(['Atrial Fibrillation'])])
print(predictions_df)
