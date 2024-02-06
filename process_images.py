
import sys
import json
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
from matplotlib import pyplot as plt
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import logging
from sklearn.model_selection import train_test_split


# Receive data from Node.js through command line arguments
data_from_nodejs = json.loads(sys.argv[1])

# print(data_from_nodejs)

# Process the data (in this case, simply print it)
# for index, filename in enumerate(data_from_nodejs):
#     print(f"Image {index + 1} received - Filename: {filename}")

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def calculate_heart_rate(ecg_signal, peaks, fs):
    # Calculate the intervals between R-peaks in samples
    r_peak_intervals = np.diff(peaks)  # Differences between successive R-peak indices

    # Convert intervals to seconds
    r_peak_intervals_sec = r_peak_intervals / fs

    # Calculate heart rate: The average number of beats per minute
    average_rr_interval = np.mean(r_peak_intervals_sec)  # Average RR interval in seconds
    heart_rate_bpm = 60 / average_rr_interval  # Convert average RR interval to BPM

    return heart_rate_bpm

def classify_rhythm(rr_intervals_sec, heart_rate_bpm):
    rr_variability = calculate_rr_variability(rr_intervals_sec)
    
    # Thresholds for classification (these are simplistic and for demonstration)
    if rr_variability > 0.1 and heart_rate_bpm > 100:  # Threshold values are illustrative
        return "Possible Tachyarrhythmia"
    elif rr_variability > 0.1:
        return "Possible Atrial Fibrillation"
    else:
        return "Normal Rhythm"
    
def plot_ecg_with_annotations(ecg_signal, peaks, rhythm_classification, fs):
    plt.figure(figsize=(15, 7))
    t = np.arange(len(ecg_signal)) / fs
    plt.plot(t, ecg_signal, label='Filtered ECG')
    plt.plot(peaks/fs, ecg_signal[peaks], "x", label='R-peaks')
    plt.title(f'ECG Signal with Rhythm Classification: {rhythm_classification}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

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

def find_t_waves(ecg_signal, s_points, fs):
    t_waves = []
    t_wave_window = int(0.3 * fs)  # 300 ms after S point, adjust as needed
    
    for s_point in s_points:
        # Search for T wave in the window after the S point
        search_region = ecg_signal[s_point:min(len(ecg_signal), s_point + t_wave_window)]
        if len(search_region) == 0:
            continue
        t_peak = np.argmax(search_region) + s_point
        t_waves.append(t_peak)
    
    return np.array(t_waves)

def is_regular_intervals(intervals, threshold=0.1):
    """Check if intervals are regular within a threshold percentage of the mean interval."""
    mean_interval = np.mean(intervals)
    return np.all(np.abs(intervals - mean_interval) < mean_interval * threshold)


def check_p_qrs_t_relationship(p_waves, q_points, r_peaks, s_points, t_waves):
    """Check if each P wave is followed by a QRS and each QRS is followed by a T wave."""
    if len(p_waves) != len(r_peaks) or len(r_peaks) != len(t_waves):
        return False  # Different counts indicate abnormal rhythm
    
    for p, q, r, s, t in zip(p_waves, q_points, r_peaks, s_points, t_waves):
        if not (p < q < r < s < t):
            return False  # Sequence order is incorrect
    
    return True

def conclude_rhythm(p_wave_regularity, qrs_regularity, t_wave_regularity, p_qrs_t_sequence, rr_variability):
    if p_wave_regularity and qrs_regularity and t_wave_regularity and p_qrs_t_sequence:
        if rr_variability < 0.1:
            return "Normal Sinus Rhythm"
        else:
            return "Irregular Sinus Rhythm"
    elif not p_qrs_t_sequence:
        return "Atrial Fibrillation or other Arrhythmia"
    else:
        return "Unclassified Arrhythmia"

def plot_ecg_with_detailed_annotations(ecg_signal, peaks, p_waves, q_points, s_points, t_waves, rhythm_conclusion, fs):
    """
    Parameters:
    - ecg_signal: The filtered ECG signal.
    - peaks: Indices of detected R peaks.
    - p_waves: Indices of detected P waves.
    - q_points, s_points: Indices of detected Q and S points of the QRS complex.
    - t_waves: Indices of detected T waves.
    - rhythm_conclusion: The concluded rhythm analysis.
    - fs: Sampling frequency of the ECG signal.
    """
    plt.figure(figsize=(20, 10))
    t = np.arange(len(ecg_signal)) / fs  # Create a time axis in seconds
    
    # Plot the ECG signal
    plt.plot(t, ecg_signal, label="Filtered ECG", color='black')
    
    # Plot annotations for R peaks, P waves, Q points, S points, and T waves
    plt.plot(peaks/fs, ecg_signal[peaks], "rx", label="R peaks", markersize=8)
    plt.plot(p_waves/fs, ecg_signal[p_waves], "bo", label="P waves", markersize=6)
    plt.plot(q_points/fs, ecg_signal[q_points], "g<", label="Q points", markersize=6)
    plt.plot(s_points/fs, ecg_signal[s_points], "g>", label="S points", markersize=6)
    plt.plot(t_waves/fs, ecg_signal[t_waves], "mv", label="T waves", markersize=6)
    
    # Set the title to include the rhythm conclusion
    plt.title(f'ECG Signal with Annotations - {rhythm_conclusion}', fontsize=16)
    
    # Label axes
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)
    
    # Add a legend
    plt.legend(loc='upper right')
    
    # Show grid
    plt.grid(True)
    
    # Display the plot
    plt.show()

def augment_ecg_signal(signal, noise_level=0.01, shift_max=0.1, scaling_factor=1.2):
    """
    Augment the ECG signal by adding Gaussian noise, shifting, and scaling.
    
    Parameters:
    - signal: Original ECG signal.
    - noise_level: Standard deviation of Gaussian noise to be added.
    - shift_max: Maximum fraction of the signal length by which to shift the signal.
    - scaling_factor: Factor by which to scale the signal amplitude.
    
    Returns:
    - Augmented ECG signal.
    """
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, signal.shape)
    signal_noisy = signal + noise
    
    # Randomly shift the signal
    shift = np.random.randint(-int(shift_max * len(signal)), int(shift_max * len(signal)))
    signal_shifted = np.roll(signal_noisy, shift)
    
    # Scale the signal
    signal_scaled = signal_shifted * scaling_factor
    
    return signal_scaled

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
    
def calculate_rr_variability(rr_intervals):
    # Calculate the standard deviation of RR intervals
    rr_std = np.std(rr_intervals)
    return rr_std

# Correct the record_path to navigate up two directories and then into the Person_01 directory
record_path = "ecg-id-database-1.0.0/Person_01/rec_1"
# Load the waveform data and header information
record = wfdb.rdrecord(record_path)

# Load the annotations
annotation = wfdb.rdann(record_path, 'atr')

# Accessing the signal
ecg_signal = record.p_signal

# Accessing annotations
annotations = annotation.symbol
annotation_indices = annotation.sample

# Accessing header information
num_leads = record.n_sig  # Number of signals (leads)
sampling_frequency = record.fs  # Sampling frequency
signal_length = record.sig_len  # Length of the signal
lead_names = record.sig_name  # Names of the leads

# Output the information
print(f"Signal shape: {ecg_signal.shape}")
print(f"Sampling frequency: {sampling_frequency} Hz")
print(f"Number of leads: {num_leads}")
print(f"Lead names: {lead_names}")
print(f"Signal length: {signal_length} samples")
print(f"First few annotations: {annotations[:10]}")
print(f"Annotation indices: {annotation_indices[:10]}")


# Normalize the signal amplitude to have zero mean and unit variance
ecg_signal_normalized = (ecg_signal - np.mean(ecg_signal, axis=0)) / np.std(ecg_signal, axis=0)

# Apply the filter
fs = sampling_frequency  # Sampling frequency from your ECG data
lowcut = 0.5  # Low cut frequency in Hz
highcut = 50.0  # High cut frequency in Hz
ecg_filtered = bandpass_filter(ecg_signal_normalized[:,0], lowcut, highcut, fs)
print(ecg_filtered)


# Detect R-peaks
peaks, _ = find_peaks(ecg_filtered, height=0.5, distance=150)  # Adjust parameters as necessary

# Calculate heart rate
heart_rate_bpm = calculate_heart_rate(ecg_filtered, peaks, fs)

# Output the average heart rate
print(f'Average Heart Rate: {heart_rate_bpm:.2f} BPM')

# Optional: Plot the filtered ECG signal and detected R-peaks to verify correctness
plt.figure(figsize=(12, 6))
plt.plot(ecg_filtered, label='Filtered ECG')
plt.plot(peaks, ecg_filtered[peaks], "x", label='Detected R-peaks')
plt.title('R-peaks in ECG Signal')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend()
plt.show()


# Calculate RR intervals in seconds
rr_intervals_sec = np.diff(peaks) / fs

# Calculate RR variability
rr_variability = calculate_rr_variability(rr_intervals_sec)

# A high variability in RR intervals might indicate AFib or other arrhythmias
print(f'RR Interval Variability (std): {rr_variability:.2f}')

rhythm_classification = classify_rhythm(rr_intervals_sec, heart_rate_bpm)
print(f'Rhythm Classification_1: {rhythm_classification}')

plot_ecg_with_annotations(ecg_filtered, peaks, rhythm_classification, fs)

# Detect P waves
p_waves = find_p_waves(ecg_filtered, peaks, fs)

# Detect QRS complexes (Q and S points)
q_points, s_points = find_qrs_complexes(ecg_filtered, peaks, fs)

# Detect T waves
t_waves = find_t_waves(ecg_filtered, s_points, fs)

p_to_r_intervals = peaks - p_waves
r_to_t_intervals = t_waves - peaks

p_wave_regularity = is_regular_intervals(p_to_r_intervals)
qrs_regularity = is_regular_intervals(np.diff(peaks))
t_wave_regularity = is_regular_intervals(r_to_t_intervals)


p_qrs_t_sequence = check_p_qrs_t_relationship(p_waves, q_points, peaks, s_points, t_waves)

rhythm_conclusion = conclude_rhythm(p_wave_regularity, qrs_regularity, t_wave_regularity, p_qrs_t_sequence, rr_variability)
print(f'Rhythm Conclusion: {rhythm_conclusion}')

plot_ecg_with_detailed_annotations(ecg_filtered, peaks, p_waves, q_points, s_points, t_waves, rhythm_conclusion, fs)


# Calculate QRS widths in milliseconds
qrs_widths_ms = (s_points - q_points) / fs * 1000

# Determine P wave presence before each QRS
p_waves_presence = len(p_waves) == len(peaks) and all(p < r for p, r in zip(p_waves, peaks))

# Infer anatomic location
anatomic_location = infer_anatomic_location(qrs_widths_ms, p_waves_presence, rhythm_classification)
print(f'Inferred Anatomic Location: {anatomic_location}')



# Assuming ecg_filtered is your filtered ECG signal and is an array-like structure
# Create a time array t that corresponds to the length of your ECG signal

fs = 500 # Replace with your actual sampling frequency
t = np.arange(len(ecg_filtered)) / fs

# Plot the filtered ECG signal
plt.figure(figsize=(12, 6))
plt.plot(t, ecg_filtered, label='Filtered ECG Signal')
plt.title('Filtered ECG Signal')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# Define the window size around the annotation index
pre_event_window_size = 100  # 200 ms before the event
post_event_window_size = 100  # 200 ms after the event

segments = []
labels = []

for idx, symbol in zip(annotation_indices, annotations):
    # Ensure the window does not go beyond the signal boundaries
    if idx > pre_event_window_size and idx < len(ecg_signal) - post_event_window_size:
        segment = ecg_signal[(idx-pre_event_window_size):(idx+post_event_window_size), :]
        segments.append(segment)
        # Assign labels based on your criteria, e.g., 0 for 'N', 1 for others
        label = 0 if symbol == 'N' else 1
        labels.append(label)

# Assuming 'segments' is a list of your data and 'labels' is a list of your labels
segments_array = np.array(segments)
labels_array = np.array(labels)

# Augment each segment in the segments_array
augmented_segments = np.array([augment_ecg_signal(segment.flatten()) for segment in segments_array])

# Reshape augmented_segments to match the input shape expected by the CNN
n_samples = augmented_segments.shape[0]
augmented_segments_reshaped = augmented_segments.reshape(n_samples, 200, 2, 1)

# Split the augmented data into training and testing sets
X_train_aug, X_test_aug, y_train, y_test = train_test_split(augmented_segments_reshaped, labels_array, test_size=0.2, random_state=42)

# reshaping given your initial data dimensions and desired 2D CNN input
X_train_reshaped = X_train_aug.reshape((X_train_aug.shape[0], 200, 2, 1))
X_test_reshaped = X_test_aug.reshape((X_test_aug.shape[0], 200, 2, 1))

print("X_train_reshaped shape:", X_train_reshaped.shape)
print("X_test_reshaped shape:", X_test_reshaped.shape)


# Define the L2 regularization penalty
l2_penalty = 0.001  # This is a hyperparameter you can tune

# Configure the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=5,          # Number of epochs to wait after min has been hit
    verbose=1,           # Print messages when stopping
    mode='min',          # The direction is automatically inferred if not set
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
)
# Define the 2D CNN model with L2 regularization
model = Sequential([
    Conv2D(32, (3, 1), activation='relu', input_shape=(200, 2, 1), kernel_regularizer=l2(l2_penalty)),  # Added L2 regularization
    MaxPooling2D((2, 1)),
    Conv2D(64, (3, 1), activation='relu', kernel_regularizer=l2(l2_penalty)),  # Added L2 regularization
    MaxPooling2D((2, 1)),
    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(l2_penalty)),  # Added L2 regularization
    Dropout(0.5),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_penalty))  # Added L2 regularization
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Use 'categorical_crossentropy' for multi-class
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=10, validation_split=0.2, batch_size=32, callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test_reshaped, y_test)
print(f'Test accuracy: {test_acc}')

# Create a dictionary with the data
output_data = {
    "heart_rate_bpm": heart_rate_bpm,
    "rhythm_classification": rhythm_classification,
    "inferred_anatomic_location": anatomic_location
}


print('---JSON_START---')
print(json.dumps(output_data))
print('---JSON_END---')
