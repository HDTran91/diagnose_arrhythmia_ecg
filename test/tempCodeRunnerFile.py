import matplotlib.pyplot as plt

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

print(segments)
print(labels)

import numpy as np
from sklearn.model_selection import train_test_split

# Assuming 'segments' is a list of your data and 'labels' is a list of your labels
segments_array = np.array(segments)
labels_array = np.array(labels)

# Now, segments_array and labels_array are NumPy arrays and have the .shape attribute
print("Segments shape:", segments_array.shape)
print("Labels shape:", labels_array.shape)

X_train, X_test, y_train, y_test = train_test_split(segments_array, labels_array, test_size=0.2, random_state=42)

# print shapes and further reshape if needed for CNN
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# reshaping given your initial data dimensions and desired 2D CNN input

X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

print("X_train_reshaped shape:", X_train_reshaped.shape)
print("X_test_reshaped shape:", X_test_reshaped.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)  # Suppress warnings

# Define the 2D CNN model
model = Sequential([
    Conv2D(32, (3, 1), activation='relu', input_shape=(200, 2, 1)),  # Adjusted kernel size
    MaxPooling2D((2, 1)),
    Conv2D(64, (3, 1), activation='relu'),  # Adjusted kernel size
    MaxPooling2D((2, 1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Use 'categorical_crossentropy' for multi-class
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=10, validation_split=0.2, batch_size=32)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test_reshaped, y_test)
print(f'Test accuracy: {test_acc}')
