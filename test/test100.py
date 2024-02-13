import wfdb
import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import wfdb.processing as wp
from scipy.signal import find_peaks

# Function to create a band-pass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    # Nyquist frequency is highest frequency that can be accurately represented when a continuous signal is sampled to produce a discrete signal.
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to apply the band-pass filter to a signal
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, signal_length=0):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, padlen=signal_length-1)
    return y

def preprocess_signal(signal, fs=100, signal_length=0):
    """
    Preprocesses the signal by handling NaN, Inf, and checking for zero variance.

    Parameters:
    - signal: The signal array to be preprocessed.

    Returns:
    - Preprocessed signal.
    """
    # Replace NaN and Inf with zero (or another strategy as needed)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure there's a check for an empty signal or add a default handling mechanism
    if signal.size == 0:
        return np.zeros(1)  # Adjust as needed

    # Apply band-pass filter
    signal = butter_bandpass_filter(signal, 0.5, 40, fs=fs, signal_length=signal_length)

    return signal

def normalize_signal(signal, range_min=0, range_max=1, fs=100, signal_length=0):
    """
    Normalizes the signal to a given range, with safety checks for zero variance.

    Parameters:
    - signal: The signal array to be normalized.
    - range_min: The minimum value of the desired normalization range.
    - range_max: The maximum value of the desired normalization range.

    Returns:
    - Normalized signal.
    """
    signal = preprocess_signal(signal, fs=fs, signal_length=signal_length)
    X_min, X_max = np.min(signal), np.max(signal)

    # Check for zero variance
    if X_max == X_min:
        return np.full(signal.shape, range_min)

    # Normalize
    normalized_signal = range_min + (signal - X_min) * (range_max - range_min) / (X_max - X_min)
    return normalized_signal


def calculate_heart_rate(peaks, fs):
    if len(peaks) < 2:
        return np.nan
    r_peak_intervals = np.diff(peaks)
    r_peak_intervals_sec = r_peak_intervals / fs
    if r_peak_intervals_sec.size == 0:
        return np.nan
    average_rr_interval = np.mean(r_peak_intervals_sec)
    heart_rate_bpm = 60 / average_rr_interval
    return heart_rate_bpm


def detect_r_peaks(ecg_lead, fs):
    peak_height = np.max(ecg_lead) * 0.8
    min_distance = int(fs * 0.2)    
    peaks, _ = find_peaks(ecg_lead, height=peak_height, distance=min_distance)
    return peaks

def calculate_rr_variability(rr_intervals):
    # Calculate the standard deviation of RR intervals
    rr_std = np.std(rr_intervals)
    return rr_std

def classify_rhythm(rr_intervals_sec, heart_rate_bpm):
    rr_variability = calculate_rr_variability(rr_intervals_sec)
    
    # Thresholds for classification (these are simplistic and for demonstration)
    if rr_variability > 0.1 and heart_rate_bpm > 100:  # Threshold values are illustrative
        return "Possible Tachyarrhythmia"
    elif rr_variability > 0.1:
        return "Possible Atrial Fibrillation"
    else:
        return "Normal Rhythm"

def find_qrs_complexes(ecg_signal, r_peaks, fs):
    q_points = []
    s_points = []
    qrs_width = int(0.1 * fs)  # 100 ms around R peak, adjust based on data
    
    for r_peak in r_peaks:
        # Define search regions for Q and S points around R peak
        q_search_region = ecg_signal[max(0, r_peak - qrs_width):r_peak]
        s_search_region = ecg_signal[r_peak:min(len(ecg_signal), r_peak + qrs_width)]
        
        # Find Q point as minimum before R peak
        if len(q_search_region) > 0:
            q_point = np.argmin(q_search_region) + max(0, r_peak - qrs_width)
            q_points.append(q_point)
        
        # Find S point as minimum after R peak
        if len(s_search_region) > 0:
            s_point = np.argmin(s_search_region) + r_peak
            s_points.append(s_point)
    
    return np.array(q_points), np.array(s_points)

def find_p_waves(ecg_signal, r_peaks, fs):
    p_waves = []
    p_wave_window = int(0.2 * fs)  # 200 ms window before R peak, adjust as needed
    
    for r_peak in r_peaks:
        # Search for P wave in the window before the R peak
        search_region = ecg_signal[max(0, r_peak - p_wave_window):r_peak]
        if len(search_region) == 0:
            continue
        p_peak = np.argmax(search_region) + max(0, r_peak - p_wave_window)
        p_waves.append(p_peak)
    
    return np.array(p_waves)

def infer_anatomic_location(qrs_widths, p_waves_presence, rhythm_classification):
    """
    Infer the anatomic location of arrhythmias based on QRS width, P wave presence, and rhythm classification.

    Parameters:
    - qrs_widths: Array of QRS complex widths in milliseconds.
    - p_waves_presence: Boolean indicating whether P waves are regularly present before each QRS complex.
    - rhythm_classification: Classification of the rhythm as normal, atrial fibrillation, etc.

    Returns:
    - A string suggesting the possible anatomic location of the arrhythmia.
    """
    mean_qrs_width = np.mean(qrs_widths)

    if not p_waves_presence and rhythm_classification == "Possible Atrial Fibrillation":
        return "Atrial origin suspected due to absence of P waves and irregular rhythm."
    elif mean_qrs_width > 120:
        return "Ventricular origin suspected due to wide QRS complexes."
    elif p_waves_presence and mean_qrs_width <= 120:
        return "Supraventricular origin suspected with normal atrial activity."
    else:
        return "Specific anatomic location unclear from available data."
    

# Directory where your ECG files are stored
ecg_directory = "small-data/records100/00000"

record_names = sorted(set(os.path.splitext(file)[0] for file in os.listdir(ecg_directory) if file.endswith('.hea')))

# Process each record
for record_name in record_names:
    record_path = os.path.join(ecg_directory, record_name)

    record = wfdb.rdrecord(record_path)
    ecg_signal = record.p_signal
    sampling_frequency = record.fs
    signal_length = record.sig_len
    lead_names = record.sig_name  # Get names of the leads

 
    
    # Initialize lists to collect data for analysis
    heart_rates = []
    rhythms = []
    annotation_locations= []

    for i, lead_name in enumerate(lead_names):
        normalized_signal = normalize_signal(ecg_signal[:, i], fs=sampling_frequency, signal_length=signal_length)
        peaks = detect_r_peaks(normalized_signal, sampling_frequency)
        rr_intervals = np.diff(peaks) / sampling_frequency  # R-R intervals in seconds

        p_waves = find_p_waves(normalized_signal, peaks, sampling_frequency)
        # Detect QRS complexes (Q and S points)
        q_points, s_points = find_qrs_complexes(normalized_signal, peaks, sampling_frequency)
        # Calculate QRS widths in milliseconds
        qrs_widths_ms = (s_points - q_points) / sampling_frequency * 1000
        # Determine P wave presence before each QRS
        p_waves_presence = len(p_waves) == len(peaks) and all(p < r for p, r in zip(p_waves, peaks))

        if len(rr_intervals) > 0:  # Ensure there are enough R-R intervals to calculate heart rate
            heart_rate_bpm = calculate_heart_rate(peaks, sampling_frequency)
            heart_rates.append(heart_rate_bpm)  # Collect heart rate

            # Classify rhythm for the lead
            rhythm = classify_rhythm(rr_intervals, heart_rate_bpm)
            rhythms.append(rhythm)  # Collect rhythm classification

            # Infer annotation location
            annotation_location = infer_anatomic_location(qrs_widths_ms, p_waves_presence, rhythm)
            annotation_locations.append(annotation_location)
            # print(f'Annotation Location: {anatomic_location}')
            
        else:
            heart_rate_bpm = np.nan  # Use NaN to indicate that heart rate couldn't be calculated
            rhythm = "Undetermined"  # Indicate that rhythm couldn't be determined
        
        # print(f'{lead_name}: Heart Rate: {heart_rate_bpm:.2f} BPM, Rhythm: {rhythm}')

    # Assuming the final rhythm classification considers the most frequently occurring classification among all leads
    from collections import Counter
    if rhythms:  # Ensure the list is not empty
        rhythm_counts = Counter(rhythms)
        most_common_rhythm = rhythm_counts.most_common(1)[0][0]  # Get the most common rhythm classification
    else:
        most_common_rhythm = "Undetermined"

    # Assuming the average heart rate is used for overall assessment
    if heart_rates:  # Ensure the list is not empty
        average_heart_rate = np.mean([hr for hr in heart_rates if not np.isnan(hr)])
    else:
        average_heart_rate = np.nan

    if annotation_locations:  # Ensure the list is not empty
        annotation_counts = Counter(annotation_locations)
        most_common_annotation_location = annotation_counts.most_common(1)[0][0]
        # print(annotation_counts)
    else:
        average_heart_rate = np.nan
    print(f'Final Rhythm Classification for the record {record_name}: {most_common_rhythm}, with an Average Heart Rate: {average_heart_rate:.2f} BPM and annotation location: {most_common_annotation_location}',)

    

    