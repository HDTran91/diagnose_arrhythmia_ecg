import numpy as np
from scipy.signal import find_peaks


def extract_normal_rhythm_features(out, sampling_frequency):
    # Calculate Heart Rate (HR)
    heart_rate = np.mean(out['heart_rate'])

    # Calculate P Wave Amplitude and Duration
    p_wave_amplitude = out['templates_ts'][0]  # Assuming P wave is the first template
    p_wave_duration = out['templates'][0].shape[0] / sampling_frequency

    # Calculate QRS Complex Duration from QRS peaks
    qrs_peaks = out['rpeaks']
    qrs_duration = np.mean(np.diff(qrs_peaks)) / sampling_frequency

    # Calculate RR Interval (as a substitute for PR Interval)
    rr_intervals = np.diff(qrs_peaks) / sampling_frequency
    pr_interval = np.mean(rr_intervals)

    # Find the last QRS peak index
    last_qrs_peak_index = qrs_peaks[-1]

    # Convert the last QRS peak index to seconds using the sampling frequency
    last_qrs_peak_time = last_qrs_peak_index / sampling_frequency

    # Find all the timestamps greater than or equal to the last QRS peak time
    timestamps_after_last_qrs_peak = out['ts'][out['ts'] >= last_qrs_peak_time]

    # Calculate the QT interval
    qt_interval = np.mean(timestamps_after_last_qrs_peak) - last_qrs_peak_time
    # Calculate QTc Interval using Bazett's formula if both QT and RR intervals are available
    if not np.isnan(qt_interval) and not np.isnan(pr_interval) and pr_interval != 0:
        qt_c = qt_interval / np.sqrt(pr_interval/1000)
    else:
        qt_c = np.nan

    return heart_rate, p_wave_amplitude, p_wave_duration, qrs_duration, pr_interval, qt_c

# Extracted features for Atrial Flutter
def extract_atrial_flutter_features(out):
    # Count F Waves
    f_wave_count = len(out['templates']) - 1  # Assuming F waves are the remaining templates
    f_wave_frequency = f_wave_count * 60 / (out['ts'][-1] - out['ts'][0])

    # Measure Atrial Rate
    atrial_rate = np.mean(out['heart_rate'])

    return f_wave_frequency, atrial_rate

# Extracted features for Atrial Fibrillation
def extract_atrial_fibrillation_features(out, sampling_frequency):
    # Determine Irregular Rhythm
    rr_intervals = np.diff(out['rpeaks']) / sampling_frequency
    irregular_rhythm = np.std(rr_intervals) > 0.1  # Standard deviation of RR intervals

    # Look for Absence of P Waves
    p_wave_presence = not np.all(np.isnan(out['templates'][0]))

    # Measure RR Interval Variability
    rr_interval_variability = np.std(rr_intervals)

    return irregular_rhythm, p_wave_presence, rr_interval_variability

# Extracted features for Sinus Tachycardia
def extract_sinus_tachycardia_features(out):
    # Evaluate Increased Heart Rate
    increased_heart_rate = np.mean(out['heart_rate']) > 100

    # Check Normal P Wave Morphology
    p_wave_normal_morphology = not np.all(np.isnan(out['templates'][0]))

    return increased_heart_rate, p_wave_normal_morphology





