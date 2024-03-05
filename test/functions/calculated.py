import numpy as np


def extract_heart_rate(peaks, sampling_frequency):
    
    
    # Calculate R-R intervals (differences between consecutive R peaks)
    rr_intervals = np.diff(peaks) / sampling_frequency
    
    # Calculate mean heart rate
    if len(rr_intervals) > 0:
        mean_hr = 60 / np.mean(rr_intervals)
    else:
        mean_hr = 0  # Placeholder, in case no peaks are detected
    
    return mean_hr

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
