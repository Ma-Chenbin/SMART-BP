import numpy as np
from scipy import signal
from heartpy import hrv


def extract_morphological_features(ppg_signal):
    # Preprocess the PPG signal if required (e.g., filtering)

    # Extract statistical features
    max_value = np.max(ppg_signal)
    min_value = np.min(ppg_signal)
    range_value = max_value - min_value
    mean_value = np.mean(ppg_signal)
    median_value = np.median(ppg_signal)
    variance = np.var(ppg_signal)
    skewness = signal.sks(ppg_signal)
    kurtosis = signal.kurtosis(ppg_signal)

    # Extract pulse rate variability
    peaks, _ = signal.find_peaks(ppg_signal, height=0.1 * range_value)
    ppi = np.diff(peaks)
    hrv_time_domain = hrv.time_domain_features(ppi)
    hrv_frequency_domain = hrv.frequency_domain_features(ppi)
    hrv_nonlinear = hrv.nonlinear_features(ppi)

    # Combine all features into a list
    features = [max_value, min_value, range_value, mean_value, median_value,
                variance, skewness, kurtosis]
    features += list(hrv_time_domain.values())
    features += list(hrv_frequency_domain.values())
    features += list(hrv_nonlinear.values())

    return features
