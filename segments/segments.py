import numpy as np
from scipy.signal import resample, find_peaks



# Fixed-length based segmentation
def fixed_length_segmentation(ppg_signal, segment_length=125*8, slide_distance=125*3):
    segments = []
    signal_length = len(ppg_signal)
    start = 0
    end = start + segment_length

    while end <= signal_length:
        segment = ppg_signal[start:end]
        segments.append(segment)
        start += slide_distance
        end += slide_distance

    return np.array(segments)



# Cycle-based segmentation with cycle detection
def cycle_based_segmentation(ppg_signal, fs, resample_length=125*8, slide_cycle=1):
    # Detect peaks in the PPG signal to find the onsets of each cardiac cycle
    peak_indices, _ = find_peaks(ppg_signal, height=0.5 * np.max(ppg_signal), distance=fs * 0.25)
    cycle_lengths = np.diff(peak_indices)

    # Find the median cycle length
    median_cycle_length = int(np.median(cycle_lengths))

    # Segment the PPG signal based on cycle lengths
    segments = []

    for i in range(len(cycle_lengths)):
        start = peak_indices[i]
        end = peak_indices[i + slide_cycle]

        # Resample the segment to a fixed length
        segment = ppg_signal[start:end]
        resampled_segment = resample(segment, resample_length)

        # Pad or truncate the resampled segment to match the median cycle length
        current_cycle_length = end - start
        if current_cycle_length > median_cycle_length:
            resampled_segment = resampled_segment[:median_cycle_length]
        elif current_cycle_length < median_cycle_length:
            padding = median_cycle_length - current_cycle_length
            resampled_segment = np.pad(resampled_segment, (0, padding), mode='constant')

        # Add the resampled segment to the list of segments
        segments.append(resampled_segment)

    # Slide the segments by a specified distance to create overlapping or non-overlapping segments
    output_segments = []

    for i in range(len(segments) - resample_length + 1):
        segment = np.array(segments[i:i + resample_length])
        output_segments.append(segment)

    return np.array(output_segments)

