import numpy as np
from scipy.signal import find_peaks, correlate


def phase_align(ppg_signal, bp_signal):
    """
    Phase align the PPG and BP signals aligned_ppg
    aligned_bp = phase_align(ppg_signal, bp_signal)
    """
    # Find the peak indices of the PPG and BP signals
    ppg_peaks, _ = find_peaks(ppg_signal)
    bp_peaks, _ = find_peaks(bp_signal)

    # Compute the cross-correlation of the PPG and BP signals
    cross_corr = correlate(ppg_signal, bp_signal, mode='same')

    # Find the index of the maximum value in the cross-correlation function
    max_index = np.argmax(cross_corr)

    # Determine the phase difference (in samples) between the PPG and BP signals
    ppg_offset = max_index - len(bp_signal) // 2

    # Align the PPG and BP signals based on the phase difference
    if ppg_offset > 0:
        aligned_ppg = np.concatenate([ppg_signal[ppg_offset:], np.zeros(ppg_offset)])
        aligned_bp = np.copy(bp_signal)
    elif ppg_offset < 0:
        aligned_ppg = np.copy(ppg_signal)
        aligned_bp = np.concatenate([bp_signal[-ppg_offset:], np.zeros(-ppg_offset)])
    else:
        aligned_ppg = np.copy(ppg_signal)
        aligned_bp = np.copy(bp_signal)

    return aligned_ppg, aligned_bp
