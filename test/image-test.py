import cv2
import numpy as np
import json
import matplotlib.pyplot as plt


# Load the image
image_path = 'small-data/image-data/image_4.png'  # Use the correct path for the image you've uploaded

# Load the image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded properly
if image is None:
    raise ValueError(f"Error: The image at {image_path} could not be loaded.")

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Use thresholding to isolate the ECG trace
_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

# Apply edge detection to find the contours of the ECG line
edges = cv2.Canny(thresh, 50, 150)

# Find contours in the edged image
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Assuming the largest contour corresponds to the ECG trace
contours = sorted(contours, key=cv2.contourArea, reverse=True)

if contours:
    ecg_contour = contours[0]
    

    # Extract the contour points
    contour_points = ecg_contour.squeeze()
    # Since ECG data is one-dimensional (time series), we need to choose one of the axes, x or y
    # For a typical ECG trace that's horizontal, we'd analyze the y-values
    ecg_signal = contour_points[:, 1]  # Assuming the ECG trace is oriented horizontally

    # Invert the signal values if necessary, because in images, the y-axis is often inverted
    ecg_signal = np.max(ecg_signal) - ecg_signal

    # Normalize the ECG signal
    ecg_signal_normalized = (ecg_signal - np.min(ecg_signal)) / (np.max(ecg_signal) - np.min(ecg_signal))
    
    # # Convert numpy array to list, as numpy arrays are not JSON serializable
    # ecg_signal_list = ecg_signal.tolist()

    # # Convert the list to a JSON-formatted string
    # ecg_signal_json = json.dumps(ecg_signal_list)
    print("ecg_signal: ", ecg_signal_normalized)
    # Now ecg_signal is a numerical array representing the ECG signal
    # Generate x values based on the length of the ECG signal
else:
    raise ValueError("No contours found in the image.")


