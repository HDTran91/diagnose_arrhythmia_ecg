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
from test.synthetic_test_model.synthetic_test_generator import generate_synthetic_data
from test.functions.preprocessing import map_scp_to_rhythm, normalize_signals
from test.functions.preprocessing import preprocess_population_data
from test.functions.preprocessing import extract_record_name_from_path
from test.functions.preprocessing import preprocess_signal_for_p_waves
from test.functions.calculated import extract_heart_rate
from test.functions.calculated import standard_deviation_rr_interval
from test.functions.calculated import classify_rhythm

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

# Apply the rhythm mapping transformation to the scp_codes column in the new dataset
combined_df['rhythm_classification'] = combined_df['scp_codes'].apply(map_scp_to_rhythm)

# Display the first few rows of the new rhythm_classification column to verify the transformation
# print(combined_df[['scp_codes', 'rhythm_classification']].head(30))

# Filtering for records with 'Normal' or 'Atrial Flutter' rhythm classifications
filtered_df = combined_df[combined_df['rhythm_classification'].isin(['Normal', 'Atrial Flutter'])]

# Displaying the filtered DataFrame
print(filtered_df.head(20))
filtered_df.to_csv("small-data/filter_data.csv", index= False)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
 
# # Gradient Boosting for Rhythm Classification

predictions_df, xgb_classifier, label_encoder = gradient_boosting_rhythm_classification(filtered_df)

print(predictions_df[predictions_df['Actual Rhythm'].isin(['Atrial Flutter'])])
# Display some of the prediction results
# print(predictions_df.head(20))

#########################################################################################################################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Create a Synthetic Test Set:

# # # Identify unique patterns from the provided DataFrame
# unique_patterns = filtered_df[['rhythm_classification']].drop_duplicates()
# print(unique_patterns)

# # # # Generate synthetic data for each unique pattern

# synthetic_test_set = generate_synthetic_data(unique_patterns, num_samples=1000)
# print(synthetic_test_set.head(5))


# ##################################################################################################
# # validate test set with Gradient Boosting
# comparison_df = evaluate_synthetic_test_set(synthetic_test_set, xgb_classifier, label_encoder)
# print(comparison_df.head(10))

# # #####################################################################################################################

