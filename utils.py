import numpy as np
from scipy.signal import find_peaks, butter, filtfilt

def low_pass_filter(data, cutoff=0.1, fs=1.0, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def check_critical_triplet(signal, station_dists=[600], time_int=20):
    peaks, _ = find_peaks(signal)
    if len(peaks) >= 3:
        times = peaks[:3]
        if max(times) - min(times) <= time_int:
            return True
    return False
