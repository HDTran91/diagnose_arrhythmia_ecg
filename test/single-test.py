
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout, BatchNormalization
from matplotlib import pyplot as plt
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay



# # Receive data from Node.js through command line arguments
# data_from_nodejs = json.loads(sys.argv[1])

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
def calculate_rr_variability(rr_intervals):
    # Calculate the standard deviation of RR intervals
    rr_std = np.std(rr_intervals)
    return rr_std

def classify_rhythm(rr_intervals_sec, heart_rate_bpm):
    rr_variability = calculate_rr_variability(rr_intervals_sec)
    
    # Thresholds for classification (these are simplistic and for demonstration)
    if rr_variability > 0.1 and heart_rate_bpm > 100:  # Threshold values are illustrative
        return "Tachyarrhythmia"
    elif rr_variability > 0.1:
        return "Fibrillation"
    else:
        return "Normal"
    
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
    
def augment_ecg_signal(signal, noise_level=0.01, shift_max=0.1, scaling_factor=1.2):
    
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, signal.shape)
    signal_noisy = signal + noise
    
    # Randomly shift the signal
    shift = np.random.randint(-int(shift_max * len(signal)), int(shift_max * len(signal)))
    signal_shifted = np.roll(signal_noisy, shift)
    
    # Scale the signal
    signal_scaled = signal_shifted * scaling_factor
    
    return signal_scaled

# Correct the record_path to navigate up two directories and then into the Person_01 directory
record_path = "small-data/records100/00000/00004_lr"
# Load the waveform data and header information
record = wfdb.rdrecord(record_path)

# Accessing the signal
ecg_signal = record.p_signal


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



# Normalize the signal amplitude to have zero mean and unit variance
ecg_signal_normalized = (ecg_signal - np.mean(ecg_signal, axis=0)) / np.std(ecg_signal, axis=0)

# Apply the filter
fs = sampling_frequency  # Sampling frequency from your ECG data
lowcut = 0.5  # Low cut frequency in Hz
highcut = 40  # High cut frequency in Hz
ecg_filtered = bandpass_filter(ecg_signal_normalized[:,0], lowcut, highcut, fs)

peak_height = np.max(ecg_filtered) * 0.5
min_distance = int(fs * 0.2) 
# Detect R-peaks
peaks, _ = find_peaks(ecg_filtered, height=peak_height, distance=min_distance)  # Adjust parameters as necessary

# Calculate heart rate
heart_rate_bpm = calculate_heart_rate(ecg_filtered, peaks, fs)

# Output the average heart rate
print(f'Average Heart Rate: {heart_rate_bpm:.2f} BPM')

# Calculate RR intervals in seconds
rr_intervals_sec = np.diff(peaks) / fs

rhythm_classification = classify_rhythm(rr_intervals_sec, heart_rate_bpm)
print(f'Rhythm Classification_1: {rhythm_classification}')

# Detect P waves
p_waves = find_p_waves(ecg_filtered, peaks, fs)
# Detect QRS complexes (Q and S points)
q_points, s_points = find_qrs_complexes(ecg_filtered, peaks, fs)
# Calculate QRS widths in milliseconds
qrs_widths_ms = (s_points - q_points) / fs * 1000
# Determine P wave presence before each QRS
p_waves_presence = len(p_waves) == len(peaks) and all(p < r for p, r in zip(p_waves, peaks))

# Infer anatomic location
anatomic_location = infer_anatomic_location(qrs_widths_ms, p_waves_presence, rhythm_classification)
print(f'Inferred Anatomic Location: {anatomic_location}')

# Define the window size around the annotation index
pre_event_window_size = 100  # 200 ms before the event
post_event_window_size = 100  # 200 ms after the event


# Extract segments around R-peaks as before
segments = []
for idx in peaks:
    if idx > pre_event_window_size and idx < len(ecg_filtered) - post_event_window_size:
        segment = ecg_filtered[(idx-pre_event_window_size):(idx+post_event_window_size)]
        segments.append(segment)

# Apply the same label to all segments
segment_label = []

if rhythm_classification == "Normal":
    segment_label.append(0)  # 0 for 'Normal'
elif rhythm_classification == "Tachyarrhythmia":
    segment_label.append(1)  # 1 for 'Tachyarrhythmia'
elif rhythm_classification == "Fibrillation":
    segment_label.append(2)  # 2 for 'Fibrillation'
segment_labels = [segment_label for _ in range(len(segments))]

# Assuming 'segments' is a list of your data and 'labels' is a list of your labels
segments_array = np.array(segments)
labels_array = np.array(segment_labels)

# Augment each segment in the segments_array
augmented_segments = np.array([augment_ecg_signal(segment.flatten()) for segment in segments_array])

# The number of samples
n_samples = augmented_segments.shape[0]


# and needs to be reshaped to (200, 2, 1) for the CNN input:
augmented_segments_reshaped = augmented_segments.reshape(n_samples, 200, 1, 1)

# Split the augmented data into training and testing sets
X_train_aug, X_test_aug, y_train, y_test = train_test_split(
    augmented_segments_reshaped, labels_array, test_size=0.2, random_state=42, stratify=labels_array)



# reshaping given your initial data dimensions and desired 2D CNN input
X_train_reshaped = X_train_aug.reshape((X_train_aug.shape[0], 200, 1, 1))
X_test_reshaped = X_test_aug.reshape((X_test_aug.shape[0], 200, 1, 1))

print("X_train_reshaped shape:", X_train_reshaped.shape)
print("X_test_reshaped shape:", X_test_reshaped.shape)

# Convert labels to one-hot encoding for multi-class classification
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

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
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(200, 1, 1), kernel_regularizer=l2(l2_penalty)),
    BatchNormalization(),
    MaxPooling2D((2, 1)),  # Adjusted pooling to avoid reducing the single-width dimension
    Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(l2_penalty)),
    BatchNormalization(),
    MaxPooling2D((2, 1)),  # Adjusted as above
    Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(l2_penalty)),
    BatchNormalization(),
    MaxPooling2D((2, 1)),  # Adjusted as above
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(l2_penalty)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(y_train_one_hot.shape[1], activation='softmax')
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Corrected loss function
              metrics=['accuracy'])



# Train the model with the corrected settings
history = model.fit(X_train_reshaped, y_train_one_hot, epochs=30, validation_split=0.2, batch_size=32, callbacks=[early_stopping])

# Evaluate the model on the test set with one-hot encoded labels
test_loss, test_acc = model.evaluate(X_test_reshaped, y_test_one_hot)
print(f'Test accuracy: {test_acc}')

# Convert probabilities to class predictions
y_pred = model.predict(X_test_reshaped)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_one_hot, axis=1)

# Calculate Precision, Recall, and F1 Score with micro, macro, or weighted averaging
precision_micro = precision_score(y_true_classes, y_pred_classes, average='micro')
precision_macro = precision_score(y_true_classes, y_pred_classes, average='macro')
recall_micro = recall_score(y_true_classes, y_pred_classes, average='micro')
recall_macro = recall_score(y_true_classes, y_pred_classes, average='macro')
f1_micro = f1_score(y_true_classes, y_pred_classes, average='micro')
f1_macro = f1_score(y_true_classes, y_pred_classes, average='macro')

print(f'Precision (Micro): {precision_micro}')
print(f'Precision (Macro): {precision_macro}')
print(f'Recall (Micro): {recall_micro}')
print(f'Recall (Macro): {recall_macro}')
print(f'F1 Score (Micro): {f1_micro}')
print(f'F1 Score (Macro): {f1_macro}')

# Plot Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()


