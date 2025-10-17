# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:40:36 2024

@author: fakbarifar
"""

#%% Posture speed
# import numpy as np
# import matplotlib.pyplot as plt

# def calculate_posture_speed(speed_data, window_size=5, std_threshold=0.005, plot=False):
#     """
#     Calculate the mean posture speed over the entire steady phase of the resampled movement data
#     based on a low variance/stability criterion.

#     Parameters:
#     - speed_data: 1D numpy array containing the resampled speed values (e.g., 64 samples).
#     - window_size: Number of samples to consider for calculating moving standard deviation (default: 5).
#     - std_threshold: Standard deviation threshold below which the hand is considered steady (default: 0.005).

#     Returns:
#     - mean_posture_speed: The average speed during the entire detected stationary phase.
#     - steady_indices: The indices of all samples in the steady phase.
#     """

#     # Calculate the moving standard deviation over the specified window size
#     moving_std = np.array([np.std(speed_data[i:i+window_size]) for i in range(len(speed_data) - window_size + 1)])

#     # Find all consecutive segments where the moving standard deviation remains below the threshold
#     steady_indices = []
#     in_steady_phase = False

#     for i in range(len(moving_std)):
#         if moving_std[i] < std_threshold:
#             if not in_steady_phase:
#                 # Start of a new steady phase
#                 in_steady_phase = True
#             # Extend steady indices to cover the current window
#             steady_indices.extend(range(i, i + window_size))
#         else:
#             if in_steady_phase:
#                 # End of steady phase
#                 break

#     # Ensure indices are unique and sorted as they may overlap due to windowing
#     steady_indices = sorted(set(steady_indices))

#     # Calculate the mean speed over the entire detected steady phase
#     if steady_indices:
#         mean_posture_speed = np.mean(speed_data[steady_indices])
#     else:
#         # Fallback if no steady phase is detected
#         mean_posture_speed = np.mean(speed_data[speed_data < np.mean(speed_data)])
        
#     # Plot the speed data with the detected steady phase highlighted
#     if plot:
#         plt.figure(figsize=(10, 4))
#         plt.plot(speed_data, label='Speed Data', color='blue')
#         plt.scatter(steady_indices, speed_data[steady_indices], color='red', label='Detected Steady Phase')
#         plt.axhline(y=mean_posture_speed, color='green', linestyle='--', label='Mean Posture Speed')
#         plt.xlabel("Sample Index")
#         plt.ylabel("Speed")
#         plt.title("Detected Steady Phase in Speed Data")
#         plt.legend()
#         plt.show()

#     return mean_posture_speed, steady_indices

# # # Example usage:
# # speed_data_resampled = np.concatenate([np.random.rand(12) * 0.01, np.random.rand(52) * 0.5])
# # posture_speed, steady_indices = calculate_posture_speed(speed_data_resampled)

# # # Plot the speed data with the detected steady phase highlighted
# # plt.plot(speed_data_resampled, label='Speed Data')
# # if steady_indices:
# #     plt.plot(steady_indices, speed_data_resampled[steady_indices], 'r', label='Detected Steady Phase', linewidth=2)
# # plt.xlabel('Sample Index')
# # plt.ylabel('Speed')
# # plt.title('Speed Data with Complete Detected Steady Phase')
# # plt.legend()
# # plt.show()

# # print(f"Calculated Posture Speed: {posture_speed:.4f} (complete steady phase)")



import numpy as np
import matplotlib.pyplot as plt

def calculate_posture_speed(speed_data, original_length, window_size=5, std_threshold=0.005, plot=False):
    """
    Calculate the mean posture speed over the steady phase of the resampled movement data,
    where the steady phase is defined as the first 200 ms in the original sampling length.

    Parameters:
    - speed_data: 1D numpy array containing the resampled speed values (e.g., 64 samples).
    - original_length: The original length of the signal before resampling.
    - window_size: Number of samples to consider for calculating moving standard deviation (default: 5).
    - std_threshold: Standard deviation threshold below which the hand is considered steady (default: 0.005).
    - plot: Boolean indicating whether to plot the detected steady phase on the speed data (default: False).

    Returns:
    - mean_posture_speed: The average speed during the entire detected steady phase.
    - steady_indices: The indices of all samples in the steady phase.
    """
    
    # Calculate how many samples correspond to the first 200 ms in the resampled signal
    steady_phase_samples = int((200 / original_length) * len(speed_data))
    
    # Extract the steady phase data for the first 200 ms in the resampled signal
    steady_phase_data = speed_data[:steady_phase_samples]
    
    # Calculate the mean speed over this steady phase
    mean_posture_speed = np.mean(steady_phase_data)
    steady_indices = list(range(steady_phase_samples))  # indices for the steady phase

    # Plot the speed data with the detected steady phase highlighted
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(speed_data, label='Speed Data', color='blue')
        plt.scatter(steady_indices, steady_phase_data, color='red', label='Detected Steady Phase')
        plt.axhline(y=mean_posture_speed, color='green', linestyle='--', label='Mean Posture Speed')
        plt.xlabel("Sample Index")
        plt.ylabel("Speed")
        plt.title("Detected Steady Phase in Speed Data")
        plt.legend()
        plt.show()

    return mean_posture_speed, steady_indices







#%% Reaction time
import numpy as np

def calculate_target_on_index(original_length, signal_length, target_on_offset=200):
    """
    Calculate the sample index corresponding to target_on based on the original trial length.

    Parameters:
    - original_length: The total duration of the original trial in milliseconds.
    - signal_length: The length of the resampled signal in samples.
    - target_on_offset: The time offset in milliseconds after the start when target_on occurs (default: 200 ms).

    Returns:
    - start_index: The sample index of target_on within the resampled signal.
    """
    # Calculate the time per sample based on the original duration and resampled length
    time_per_sample = original_length / signal_length
    start_index = int(target_on_offset / time_per_sample)
    return start_index

def calculate_reaction_time_slope(speed_data, slope_threshold=0.01, original_length=None, target_on_offset=200):
    """
    Detect movement onset based on the slope of speed data, starting only after target_on.

    Parameters:
    - speed_data: 1D numpy array containing the resampled speed values.
    - slope_threshold: Threshold for the rate of change in speed (default: 0.01).
    - original_length: The total duration of the original trial in milliseconds (default: None).
    - target_on_offset: Time after the start when target_on occurs (default: 200 ms).

    Returns:
    - onset_index / len(speed_data): Normalized index of movement onset.
    """
    if original_length is None:
        raise ValueError("original_length must be provided for accurate target_on calculation.")

    # Calculate target_on index based on the actual length of the signal
    start_index = calculate_target_on_index(original_length, len(speed_data), target_on_offset)
    
    # Calculate the difference between consecutive speed samples (slope)
    slopes = np.diff(speed_data)

    # Find the first instance where slope exceeds the threshold, starting from start_index
    if np.argmax(slopes[start_index:] > slope_threshold)==0:
        onset_index = start_index + np.argmax(slopes[start_index:] > slope_threshold) +1
    else: 
        onset_index = start_index + np.argmax(slopes[start_index:] > slope_threshold)
    # Normalize by the total number of samples
    return onset_index / len(speed_data)

def calculate_reaction_time_cusum(speed_data, cusum_threshold=0.02):
    """
    Detect movement onset based on the CUSUM method.

    Parameters:
    - speed_data: 1D numpy array containing the resampled speed values.
    - cusum_threshold: Threshold for cumulative sum change (default: 0.02).

    Returns:
    - reaction_time: Normalized index of movement onset.
    """
    # Calculate the CUSUM for changes in speed
    mean_speed = np.mean(speed_data)
    cusum = np.cumsum(speed_data - mean_speed)

    # Detect when the cumulative sum exceeds the threshold
    onset_index = np.argmax(cusum > cusum_threshold)
    
    return onset_index / len(speed_data)

def calculate_reaction_time_moving_average(speed_data, window_size=5, ma_threshold=0.02):
    """
    Detect movement onset based on a moving average of speed data.

    Parameters:
    - speed_data: 1D numpy array containing the resampled speed values.
    - window_size: Number of samples for the moving average (default: 5).
    - ma_threshold: Speed value above which the hand is considered to have started moving (default: 0.02).

    Returns:
    - reaction_time: Normalized index of movement onset.
    """
    # Compute the moving average of the speed data
    moving_avg = np.convolve(speed_data, np.ones(window_size) / window_size, mode='valid')

    # Find the first instance where the moving average exceeds the threshold
    onset_index = np.argmax(moving_avg > ma_threshold)
    
    # Normalize by the total number of samples (considering moving average length)
    return onset_index / len(speed_data)






# import numpy as np
# import matplotlib.pyplot as plt

# # Set random seed for reproducibility
# np.random.seed(42)

# # Parameters for simulation
# stationary_length = 12   # Length of stationary phase
# movement_length = 52     # Length of movement phase
# stationary_speed = 0.01  # Max speed during stationary phase
# movement_speed = 0.5     # Max speed during movement phase

# # Create stationary phase with low speed
# stationary_phase = np.random.rand(stationary_length) * stationary_speed

# # Create movement phase with higher speed
# movement_phase = np.random.rand(movement_length) * movement_speed

# # Concatenate to form the complete speed signal
# speed_data_resampled = np.concatenate([stationary_phase, movement_phase])

# # Define methods for detecting movement onset
# def calculate_reaction_time_slope(speed_data, slope_threshold=0.01):
#     slopes = np.diff(speed_data)
#     onset_index = np.argmax(slopes > slope_threshold)
#     return onset_index / len(speed_data)

# def calculate_reaction_time_cusum(speed_data, cusum_threshold=0.02):
#     mean_speed = np.mean(speed_data)
#     cusum = np.cumsum(speed_data - mean_speed)
#     onset_index = np.argmax(cusum > cusum_threshold)
#     return onset_index / len(speed_data)

# def calculate_reaction_time_moving_average(speed_data, window_size=5, ma_threshold=0.02):
#     moving_avg = np.convolve(speed_data, np.ones(window_size) / window_size, mode='valid')
#     onset_index = np.argmax(moving_avg > ma_threshold)
#     return onset_index / len(speed_data)

# # Calculate reaction time using each method with the simulated data
# slope_threshold = 0.01
# cusum_threshold = 0.02
# ma_threshold = 0.02
# window_size = 5

# reaction_time_slope = calculate_reaction_time_slope(speed_data_resampled, slope_threshold)
# reaction_time_cusum = calculate_reaction_time_cusum(speed_data_resampled, cusum_threshold)
# reaction_time_moving_average = calculate_reaction_time_moving_average(speed_data_resampled, window_size, ma_threshold)

# # Convert reaction times to sample indices for plotting
# slope_index = int(reaction_time_slope * len(speed_data_resampled))
# cusum_index = int(reaction_time_cusum * len(speed_data_resampled))
# ma_index = int(reaction_time_moving_average * len(speed_data_resampled))

# # Plot the simulated speed data with detected movement onset points for each method
# plt.figure(figsize=(12, 6))
# plt.plot(speed_data_resampled, label='Simulated Speed Data', color='blue')
# plt.axvline(x=slope_index, color='green', linestyle='--', label='Slope Method Onset')
# plt.axvline(x=cusum_index, color='orange', linestyle='--', label='CUSUM Method Onset')
# plt.axvline(x=ma_index, color='red', linestyle='--', label='Moving Average Method Onset')
# plt.xlabel('Sample Index')
# plt.ylabel('Speed')
# plt.title('Simulated Speed Data with Movement Onset Detected by Various Methods')
# plt.legend()
# plt.show()

# Function to calculate reaction time with original lengths
def calculate_physical_reaction_time(trial, detection_method, original_length, plot=False, **kwargs):
    """
    Calculate reaction time in physical units (milliseconds) and optionally plot the speed data
    with target_on and movement onset markers.

    Parameters:
    - trial: 1D numpy array containing the resampled speed values.
    - detection_method: Function to detect reaction time (e.g., calculate_reaction_time_slope).
    - original_length: The original length of the trial in milliseconds.
    - plot: Boolean indicating whether to plot the detected movement onset (default: False).

    Returns:
    - reaction_time_physical: Reaction time in milliseconds from target_on to movement onset.
    """
    # Calculate reaction time index in the resampled trial (normalized to range 0 to 1)
    reaction_time_resampled = detection_method(trial, original_length=original_length, **kwargs)
    reaction_time_index = int(reaction_time_resampled * len(trial))  # Index in 64-sample data

    # Convert reaction time index back to original length in milliseconds
    reaction_time_physical = (reaction_time_index / len(trial)) * original_length - 200  # Subtract 200 ms for target_on

    # Plot the speed data with target_on and movement onset markers
    if plot:
        time_per_sample = original_length / len(trial)  # Convert samples to milliseconds
        time_axis = np.arange(len(trial)) * time_per_sample  # Create time axis in milliseconds

        # Calculate positions for target_on (200 ms) and movement onset
        target_on_time = 200
        movement_onset_time = reaction_time_index * time_per_sample

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, trial, label='Speed Data', color='blue')
        plt.axvline(x=target_on_time, color='purple', linestyle='--', label='Target On (200 ms)')
        plt.axvline(x=movement_onset_time, color='red', linestyle='--', label='Movement Onset')
        plt.xlabel("Time (ms)")
        plt.ylabel("Speed")
        plt.title("Speed Data with Target On and Movement Onset")
        plt.legend()
        plt.show()
        
    return reaction_time_physical

#%% speed maxima count between movement onset and offset
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks

# # Simulated resampled speed data (64 samples): initial low-speed portion then movement
# np.random.seed(42)
# stationary_length = 12
# movement_length = 52
# stationary_speed = 0.01
# movement_speed = 0.5

# stationary_phase = np.random.rand(stationary_length) * stationary_speed
# movement_phase = np.random.rand(movement_length) * movement_speed
# speed_data_resampled = np.concatenate([stationary_phase, movement_phase])

# # Define detection methods for movement onset and offset
# def calculate_movement_onset_offset_slope(speed_data, onset_threshold=0.01, offset_threshold=0.01):
#     slopes = np.diff(speed_data)
#     onset_index = np.argmax(slopes > onset_threshold)
#     offset_index = onset_index + np.argmax(slopes[onset_index:] < -offset_threshold)
#     return onset_index, offset_index

# def calculate_movement_onset_offset_cusum(speed_data, cusum_threshold=0.02):
#     mean_speed = np.mean(speed_data)
#     cusum = np.cumsum(speed_data - mean_speed)
#     onset_index = np.argmax(cusum > cusum_threshold)
#     offset_index = onset_index + np.argmax(cusum[onset_index:] < cusum_threshold)
#     return onset_index, offset_index

# def calculate_movement_onset_offset_moving_average(speed_data, window_size=5, ma_threshold=0.02):
#     moving_avg = np.convolve(speed_data, np.ones(window_size) / window_size, mode='valid')
#     onset_index = np.argmax(moving_avg > ma_threshold)
#     offset_index = onset_index + np.argmax(moving_avg[onset_index:] < ma_threshold)
#     return onset_index, offset_index

# # Define function to count speed maxima between onset and offset
# def calculate_speed_maxima_count_between_onset_offset(speed_data, detection_method, **kwargs):
#     onset_index, offset_index = detection_method(speed_data, **kwargs)
#     if offset_index <= onset_index:
#         offset_index = len(speed_data) - 1
#     movement_phase_data = speed_data[onset_index:offset_index]
#     peaks, _ = find_peaks(movement_phase_data)
#     return len(peaks), onset_index, offset_index, peaks

# # Parameters for each detection method
# slope_params = {'onset_threshold': 0.01, 'offset_threshold': 0.01}
# cusum_params = {'cusum_threshold': 0.02}
# ma_params = {'window_size': 5, 'ma_threshold': 0.02}

# # Apply the methods
# maxima_count_slope, onset_index_slope, offset_index_slope, peaks_slope = calculate_speed_maxima_count_between_onset_offset(
#     speed_data_resampled, calculate_movement_onset_offset_slope, **slope_params
# )
# maxima_count_cusum, onset_index_cusum, offset_index_cusum, peaks_cusum = calculate_speed_maxima_count_between_onset_offset(
#     speed_data_resampled, calculate_movement_onset_offset_cusum, **cusum_params
# )
# maxima_count_ma, onset_index_ma, offset_index_ma, peaks_ma = calculate_speed_maxima_count_between_onset_offset(
#     speed_data_resampled, calculate_movement_onset_offset_moving_average, **ma_params
# )

# # Visualization
# plt.figure(figsize=(15, 12))

# # Slope Method
# plt.subplot(3, 1, 1)
# plt.plot(speed_data_resampled, label='Speed Data')
# plt.axvline(onset_index_slope, color='green', linestyle='--', label='Onset (Slope)')
# plt.axvline(offset_index_slope, color='red', linestyle='--', label='Offset (Slope)')
# plt.plot(onset_index_slope + peaks_slope, speed_data_resampled[onset_index_slope + peaks_slope], 'ro', label='Local Maxima')
# plt.title(f'Slope Method - Detected Maxima Count: {maxima_count_slope}')
# plt.xlabel('Sample Index')
# plt.ylabel('Speed')
# plt.legend()

# # CUSUM Method
# plt.subplot(3, 1, 2)
# plt.plot(speed_data_resampled, label='Speed Data')
# plt.axvline(onset_index_cusum, color='green', linestyle='--', label='Onset (CUSUM)')
# plt.axvline(offset_index_cusum, color='red', linestyle='--', label='Offset (CUSUM)')
# plt.plot(onset_index_cusum + peaks_cusum, speed_data_resampled[onset_index_cusum + peaks_cusum], 'ro', label='Local Maxima')
# plt.title(f'CUSUM Method - Detected Maxima Count: {maxima_count_cusum}')
# plt.xlabel('Sample Index')
# plt.ylabel('Speed')
# plt.legend()

# # Moving Average Method
# plt.subplot(3, 1, 3)
# plt.plot(speed_data_resampled, label='Speed Data')
# plt.axvline(onset_index_ma, color='green', linestyle='--', label='Onset (Moving Average)')
# plt.axvline(offset_index_ma, color='red', linestyle='--', label='Offset (Moving Average)')
# plt.plot(onset_index_ma + peaks_ma, speed_data_resampled[onset_index_ma + peaks_ma], 'ro', label='Local Maxima')
# plt.title(f'Moving Average Method - Detected Maxima Count: {maxima_count_ma}')
# plt.xlabel('Sample Index')
# plt.ylabel('Speed')
# plt.legend()

# plt.tight_layout()
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def calculate_speed_maxima_count(speed_data, detection_method, original_length, slope_threshold=0.01, plot=False):
    """
    Calculate the count of speed maxima (peaks) between the movement onset and end of the signal.

    Parameters:
    - speed_data: 1D numpy array containing the resampled speed values.
    - detection_method: Function to detect movement onset (e.g., calculate_reaction_time_slope).
    - original_length: The total duration of the original trial in milliseconds.
    - slope_threshold: Threshold for the rate of change in speed (default: 0.01).
    - plot: Boolean indicating whether to plot the detected peaks and movement onset (default: False).

    Returns:
    - maxima_count: The count of local maxima (peaks) in the movement phase.
    - onset_index: The index of movement onset within the speed data.
    - offset_index: The index of movement offset (end of signal).
    - peaks: The indices of the detected peaks within the movement phase.
    """
    # Detect movement onset
    onset_index = int(detection_method(speed_data, slope_threshold=slope_threshold, original_length=original_length) * len(speed_data))
    
    # Set movement offset to the end of the signal
    offset_index = len(speed_data) - 1

    # Extract the movement phase data (from onset to offset)
    movement_phase_data = speed_data[onset_index:offset_index]

    # Find peaks in the movement phase
    peaks, _ = find_peaks(movement_phase_data)
    maxima_count = len(peaks)

    # Plot if requested
    if plot:
        time_per_sample = original_length / len(speed_data)  # Time per sample in ms
        time_axis = np.arange(len(speed_data)) * time_per_sample  # Time axis for the entire signal
        movement_phase_time = time_axis[onset_index:offset_index]  # Time axis for the movement phase

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, speed_data, label='Speed Data', color='blue')
        plt.axvline(x=200, color='purple', linestyle='--', label='Target On (200 ms)')
        plt.axvline(x=onset_index * time_per_sample, color='green', linestyle='--', label='Movement Onset')
        plt.axvline(x=offset_index * time_per_sample, color='red', linestyle='--', label='End of Signal (Offset)')
        plt.plot(movement_phase_time[peaks], movement_phase_data[peaks], 'ro', label='Local Maxima')
        plt.xlabel("Time (ms)")
        plt.ylabel("Speed")
        plt.title(f"Speed Data with Movement Onset, Offset, and Detected Peaks (Count: {maxima_count})")
        plt.legend()
        plt.show()

    return maxima_count, onset_index, offset_index, peaks + onset_index  # Adjust peaks to full signal index

# Example usage:
# original_length = 2000  # Example original trial length in ms (for demonstration)
# speed_data_resampled = np.concatenate([np.random.rand(12) * 0.01, np.random.rand(52) * 0.5])  # Simulated example data
# maxima_count, onset_index, offset_index, peaks = calculate_speed_maxima_count(
#     speed_data_resampled, calculate_reaction_time_slope, original_length, slope_threshold=0.01, plot=True
# )

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def calculate_significant_speed_peaks_dynamic_prominence(speed_data, detection_method, original_length, slope_threshold=0.01, plot=False):
    """
    Detect significant speed peaks (prominent peaks) between the movement onset and end of the signal,
    with prominence set as a fraction of the maximum peak in the trial.

    Parameters:
    - speed_data: 1D numpy array containing the resampled speed values.
    - detection_method: Function to detect movement onset (e.g., calculate_reaction_time_slope).
    - original_length: The total duration of the original trial in milliseconds.
    - slope_threshold: Threshold for the rate of change in speed (default: 0.01).
    - plot: Boolean indicating whether to plot the detected peaks and movement onset (default: False).

    Returns:
    - significant_peaks_count: The count of significant peaks in the movement phase.
    - onset_index: The index of movement onset within the speed data.
    - offset_index: The index of movement offset (end of signal).
    - peaks: The indices of the detected significant peaks within the movement phase.
    """
    # Calculate dynamic prominence as a fraction of the maximum peak in the trial
    dynamic_prominence = np.max(speed_data) / 50  # Adjust the denominator as needed
    
    # Detect movement onset
    onset_index = int(detection_method(speed_data, slope_threshold=slope_threshold, original_length=original_length) * len(speed_data))
    
    # Set movement offset to the end of the signal
    offset_index = len(speed_data) - 1

    # Extract the movement phase data (from onset to offset)
    movement_phase_data = speed_data[onset_index:offset_index]

    # Find prominent peaks in the movement phase using the dynamic prominence
    peaks, _ = find_peaks(movement_phase_data, prominence=dynamic_prominence)
    significant_peaks_count = len(peaks)

    # Plot if requested
    if plot:
        time_per_sample = original_length / len(speed_data)  # Time per sample in ms
        time_axis = np.arange(len(speed_data)) * time_per_sample  # Time axis for the entire signal
        movement_phase_time = time_axis[onset_index:offset_index]  # Time axis for the movement phase

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, speed_data, label='Speed Data', color='blue')
        plt.axvline(x=200, color='purple', linestyle='--', label='Target On (200 ms)')
        plt.axvline(x=onset_index * time_per_sample, color='green', linestyle='--', label='Movement Onset')
        plt.axvline(x=offset_index * time_per_sample, color='red', linestyle='--', label='End of Signal (Offset)')
        plt.plot(movement_phase_time[peaks], movement_phase_data[peaks], 'ro', label='Significant Peaks')
        plt.xlabel("Time (ms)")
        plt.ylabel("Speed")
        plt.title(f"Speed Data with Movement Onset, Offset, and Significant Peaks (Count: {significant_peaks_count})")
        plt.legend()
        plt.show()

    # Adjust peaks to the full signal index
    return significant_peaks_count, onset_index, offset_index, peaks + onset_index

# # Example usage:
# original_length = 2000  # Example original trial length in ms (for demonstration)
# speed_data_resampled = np.concatenate([np.random.rand(12) * 0.01, np.random.rand(52) * 0.5])  # Simulated example data
# significant_peaks_count, onset_index, offset_index, peaks = calculate_significant_speed_peaks_dynamic_prominence(
#     speed_data_resampled, calculate_reaction_time_slope, original_length, slope_threshold=0.01, plot=True
# )


#%% min-max speed difference
from scipy.signal import find_peaks

def calculate_min_max_speed_difference(speed_data, detection_method, **kwargs):
    """
    Calculate the mean difference between adjacent local speed minima and maxima within the movement phase.
    
    Parameters:
    - speed_data: 1D numpy array containing the resampled speed values.
    - detection_method: Function to detect movement onset and offset (Slope, CUSUM, or Moving Average).
    - kwargs: Additional parameters for the detection method (thresholds, window size, etc.).
    
    Returns:
    - mean_min_max_diff: The mean difference between adjacent local minima and maxima.
    - onset_index: The index of movement onset.
    - offset_index: The index of movement offset.
    - extrema_indices: Indices of the detected local minima and maxima.
    """
    # Detect onset and offset using the provided detection method
    onset_index, offset_index = detection_method(speed_data, **kwargs)

    # If offset is not greater than onset, set offset to the end of the data
    if offset_index <= onset_index:
        offset_index = len(speed_data) - 1

    # Extract the movement phase data
    movement_phase_data = speed_data[onset_index:offset_index]

    # Find local maxima and minima within the movement phase
    maxima, _ = find_peaks(movement_phase_data)
    minima, _ = find_peaks(-movement_phase_data)  # Invert to find minima

    # Combine and sort minima and maxima indices
    extrema_indices = np.sort(np.concatenate([maxima, minima]))

    # Calculate the differences between adjacent minima and maxima
    min_max_diffs = np.abs(np.diff(movement_phase_data[extrema_indices]))

    # Calculate the mean difference, or return zero if no differences are found
    mean_min_max_diff = np.mean(min_max_diffs) if len(min_max_diffs) > 0 else 0.0

    return mean_min_max_diff, onset_index, offset_index, extrema_indices



# # Parameters for each detection method
# slope_params = {'onset_threshold': 0.01, 'offset_threshold': 0.01}
# cusum_params = {'cusum_threshold': 0.02}
# ma_params = {'window_size': 5, 'ma_threshold': 0.02}

# # Calculate min-max speed difference using Slope method
# mean_diff_slope, onset_slope, offset_slope, extrema_slope = calculate_min_max_speed_difference(
#     speed_data_resampled, calculate_movement_onset_offset_slope, **slope_params
# )

# # Calculate min-max speed difference using CUSUM method
# mean_diff_cusum, onset_cusum, offset_cusum, extrema_cusum = calculate_min_max_speed_difference(
#     speed_data_resampled, calculate_movement_onset_offset_cusum, **cusum_params
# )

# # Calculate min-max speed difference using Moving Average method
# mean_diff_ma, onset_ma, offset_ma, extrema_ma = calculate_min_max_speed_difference(
#     speed_data_resampled, calculate_movement_onset_offset_moving_average, **ma_params
# )

# # Print the results
# print("Slope Method - Mean Min-Max Difference:", mean_diff_slope)
# print("CUSUM Method - Mean Min-Max Difference:", mean_diff_cusum)
# print("Moving Average Method - Mean Min-Max Difference:", mean_diff_ma)



import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def calculate_min_max_speed_difference_dynamic_extrema(speed_data, detection_method, original_length, slope_threshold=0.01, prominence_fraction=(1/14), plot=False):
    """
    Calculate the mean difference between adjacent prominent local minima and maxima within the movement phase,
    with dynamic prominence and optional plotting.

    Parameters:
    - speed_data: 1D numpy array containing the resampled speed values.
    - detection_method: Function to detect movement onset (e.g., calculate_reaction_time_slope).
    - original_length: The total duration of the original trial in milliseconds.
    - slope_threshold: Threshold for the rate of change in speed for movement onset detection (default: 0.01).
    - prominence_fraction: Fraction of the maximum peak value to use for prominence in extrema detection (default: 0.1).
    - plot: Boolean indicating whether to plot the detected movement onset, offset, and extrema (default: False).

    Returns:
    - mean_min_max_diff: The mean difference between adjacent prominent local minima and maxima.
    - onset_index: The index of movement onset.
    - offset_index: The index of movement offset (end of signal).
    - extrema_indices: Indices of the detected local minima and maxima.
    """
    # Calculate dynamic prominence based on the signal’s maximum value
    dynamic_prominence = np.max(speed_data) * prominence_fraction

    # Detect movement onset
    onset_index = int(detection_method(speed_data, slope_threshold=slope_threshold, original_length=original_length) * len(speed_data))
    
    # Set movement offset to the end of the signal
    offset_index = len(speed_data) - 1

    # Extract the movement phase data
    movement_phase_data = speed_data[onset_index:offset_index]

    # Find prominent local maxima and minima within the movement phase
    maxima, _ = find_peaks(movement_phase_data, prominence=dynamic_prominence)
    minima, _ = find_peaks(-movement_phase_data, prominence=dynamic_prominence)  # Invert to find minima

    # Combine and sort minima and maxima indices
    extrema_indices = np.sort(np.concatenate([maxima, minima]))

    # Calculate the differences between adjacent minima and maxima
    min_max_diffs = np.abs(np.diff(movement_phase_data[extrema_indices]))

    # Calculate the mean difference, or return zero if no differences are found
    mean_min_max_diff = np.mean(min_max_diffs) if len(min_max_diffs) > 0 else 0.0

    # Plot if requested
    if plot:
        time_per_sample = original_length / len(speed_data)  # Time per sample in ms
        time_axis = np.arange(len(speed_data)) * time_per_sample  # Time axis for the entire signal
        movement_phase_time = time_axis[onset_index:offset_index]  # Time axis for the movement phase

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, speed_data, label='Speed Data', color='blue')
        plt.axvline(x=200, color='purple', linestyle='--', label='Target On (200 ms)')
        plt.axvline(x=onset_index * time_per_sample, color='green', linestyle='--', label='Movement Onset')
        plt.axvline(x=offset_index * time_per_sample, color='red', linestyle='--', label='End of Signal (Offset)')
        plt.plot(movement_phase_time[extrema_indices], movement_phase_data[extrema_indices], 'ro', label='Prominent Extrema')
        plt.xlabel("Time (ms)")
        plt.ylabel("Speed")
        plt.title(f"Speed Data with Movement Onset, Offset, and Prominent Extrema (Mean Difference: {mean_min_max_diff:.3f})")
        plt.legend()
        plt.show()

    return mean_min_max_diff, onset_index, offset_index, extrema_indices

# Example usage:
# original_length = 2000  # Example original trial length in ms
# speed_data_resampled = np.concatenate([np.random.rand(12) * 0.01, np.random.rand(52) * 0.5])  # Simulated example data
# mean_min_max_diff, onset_index, offset_index, extrema_indices = calculate_min_max_speed_difference_dynamic_extrema(
#     speed_data_resampled, calculate_reaction_time_slope, original_length, slope_threshold=0.01, prominence_fraction=0.1, plot=True
# )


#%% movement time
def calculate_movement_time_resampled(speed_data, detection_method, **kwargs):
    """
    Calculate the movement time based on the resampled movement data using advanced onset/offset detection.
    
    Parameters:
    - speed_data: 1D numpy array containing the resampled speed values.
    - detection_method: Function to detect movement onset and offset (Slope, CUSUM, or Moving Average).
    - kwargs: Additional parameters for the detection method (thresholds, window size, etc.).
    
    Returns:
    - movement_time_normalized: The movement time as a fraction of the total samples.
    - onset_index: The index of movement onset.
    - offset_index: The index of movement offset.
    """
    # Detect onset and offset using the provided detection method
    onset_index, offset_index = detection_method(speed_data, **kwargs)
    
    # Ensure valid indices: If offset is not greater than onset, set offset to the end of the data
    if offset_index <= onset_index:
        offset_index = len(speed_data) - 1

    # Calculate movement time in terms of samples
    movement_time_samples = offset_index - onset_index
    
    # Normalize movement time by the total number of samples
    movement_time_normalized = movement_time_samples / len(speed_data)
    
    return movement_time_normalized, onset_index, offset_index

# Adjusted function to calculate physical movement time by setting offset as end of trial
def calculate_physical_movement_time(trial, detection_method, original_length, **kwargs):
    # Calculate movement onset in the resampled (64-sample) data using the detection method and params
    onset_index_resampled, _ = detection_method(trial, **kwargs)
    offset_index_resampled = len(trial) - 1  # Set offset to the end of the trial

    # Convert onset and offset to original length
    onset_physical = int((onset_index_resampled / len(trial)) * original_length)
    offset_physical = original_length  # Offset as the end of trial

    # Calculate physical movement time
    movement_time_physical = offset_physical - onset_physical
    return movement_time_physical

# import numpy as np
# import matplotlib.pyplot as plt

# # Simulated resampled speed data (64 samples): initial low-speed portion then movement
# np.random.seed(42)
# stationary_length = 12
# movement_length = 52
# stationary_speed = 0.01
# movement_speed = 0.5

# stationary_phase = np.random.rand(stationary_length) * stationary_speed
# movement_phase = np.random.rand(movement_length) * movement_speed
# speed_data_resampled = np.concatenate([stationary_phase, movement_phase])

# Define the advanced detection methods
def calculate_movement_onset_offset_slope(speed_data, onset_threshold=0.01, offset_threshold=0.01):
    slopes = np.diff(speed_data)
    onset_index = np.argmax(slopes > onset_threshold)
    offset_index = onset_index + np.argmax(slopes[onset_index:] < -offset_threshold)
    return onset_index, offset_index

def calculate_movement_onset_offset_cusum(speed_data, cusum_threshold=0.02):
    mean_speed = np.mean(speed_data)
    cusum = np.cumsum(speed_data - mean_speed)
    onset_index = np.argmax(cusum > cusum_threshold)
    offset_index = onset_index + np.argmax(cusum[onset_index:] < cusum_threshold)
    return onset_index, offset_index

def calculate_movement_onset_offset_moving_average(speed_data, window_size=5, ma_threshold=0.02):
    moving_avg = np.convolve(speed_data, np.ones(window_size) / window_size, mode='valid')
    onset_index = np.argmax(moving_avg > ma_threshold)
    offset_index = onset_index + np.argmax(moving_avg[onset_index:] < ma_threshold)
    return onset_index, offset_index

# Define movement time calculation
def calculate_movement_time_resampled(speed_data, detection_method, **kwargs):
    onset_index, offset_index = detection_method(speed_data, **kwargs)
    if offset_index <= onset_index:
        offset_index = len(speed_data) - 1
    movement_time_samples = offset_index - onset_index
    movement_time_normalized = movement_time_samples / len(speed_data)
    return movement_time_normalized, onset_index, offset_index

# # Parameters for each detection method
# slope_params = {'onset_threshold': 0.01, 'offset_threshold': 0.01}
# cusum_params = {'cusum_threshold': 0.02}
# ma_params = {'window_size': 5, 'ma_threshold': 0.02}

# # Calculate movement time and plot results for each method
# # Slope Method
# movement_time_slope, onset_slope, offset_slope = calculate_movement_time_resampled(
#     speed_data_resampled, calculate_movement_onset_offset_slope, **slope_params
# )
# # CUSUM Method
# movement_time_cusum, onset_cusum, offset_cusum = calculate_movement_time_resampled(
#     speed_data_resampled, calculate_movement_onset_offset_cusum, **cusum_params
# )
# # Moving Average Method
# movement_time_ma, onset_ma, offset_ma = calculate_movement_time_resampled(
#     speed_data_resampled, calculate_movement_onset_offset_moving_average, **ma_params
# )

# # Visualization
# plt.figure(figsize=(15, 12))

# # Slope Method
# plt.subplot(3, 1, 1)
# plt.plot(speed_data_resampled, label='Speed Data')
# plt.axvline(onset_slope, color='green', linestyle='--', label='Onset (Slope)')
# plt.axvline(offset_slope, color='red', linestyle='--', label='Offset (Slope)')
# plt.fill_between(range(onset_slope, offset_slope), speed_data_resampled[onset_slope:offset_slope], alpha=0.3)
# plt.title(f'Slope Method - Normalized Movement Time: {movement_time_slope:.2f}')
# plt.xlabel('Sample Index')
# plt.ylabel('Speed')
# plt.legend()

# # CUSUM Method
# plt.subplot(3, 1, 2)
# plt.plot(speed_data_resampled, label='Speed Data')
# plt.axvline(onset_cusum, color='green', linestyle='--', label='Onset (CUSUM)')
# plt.axvline(offset_cusum, color='red', linestyle='--', label='Offset (CUSUM)')
# plt.fill_between(range(onset_cusum, offset_cusum), speed_data_resampled[onset_cusum:offset_cusum], alpha=0.3)
# plt.title(f'CUSUM Method - Normalized Movement Time: {movement_time_cusum:.2f}')
# plt.xlabel('Sample Index')
# plt.ylabel('Speed')
# plt.legend()

# # Moving Average Method
# plt.subplot(3, 1, 3)
# plt.plot(speed_data_resampled, label='Speed Data')
# plt.axvline(onset_ma, color='green', linestyle='--', label='Onset (Moving Average)')
# plt.axvline(offset_ma, color='red', linestyle='--', label='Offset (Moving Average)')
# plt.fill_between(range(onset_ma, offset_ma), speed_data_resampled[onset_ma:offset_ma], alpha=0.3)
# plt.title(f'Moving Average Method - Normalized Movement Time: {movement_time_ma:.2f}')
# plt.xlabel('Sample Index')
# plt.ylabel('Speed')
# plt.legend()

# plt.tight_layout()
# plt.show()



import numpy as np
import matplotlib.pyplot as plt

def calculate_movement_time(speed_data, detection_method, original_length, slope_threshold=0.01, plot=False):
    """
    Calculate movement time as the duration from movement onset to movement offset,
    where offset is defined as the end of the signal.

    Parameters:
    - speed_data: 1D numpy array containing the resampled speed values.
    - detection_method: Function to detect movement onset (e.g., calculate_reaction_time_slope).
    - original_length: The total duration of the original trial in milliseconds.
    - slope_threshold: Threshold for the rate of change in speed for movement onset detection (default: 0.01).
    - plot: Boolean indicating whether to plot the detected movement onset, offset, and movement time (default: False).

    Returns:
    - movement_time: The time in milliseconds from movement onset to movement offset.
    - onset_index: The index of movement onset.
    - offset_index: The index of movement offset (end of signal).
    """
    # Detect movement onset after target-on (200 ms after the start)
    onset_index = int(detection_method(speed_data, slope_threshold=slope_threshold, original_length=original_length) * len(speed_data))
    
    # Set movement offset to the end of the signal
    offset_index = len(speed_data) - 1

    # Calculate movement time in milliseconds
    time_per_sample = original_length / len(speed_data)  # Time per sample in ms
    movement_time = (offset_index - onset_index) * time_per_sample

    # Plot if requested
    if plot:
        time_axis = np.arange(len(speed_data)) * time_per_sample  # Time axis for the entire signal

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, speed_data, label='Speed Data', color='blue')
        plt.axvline(x=200, color='purple', linestyle='--', label='Target On (200 ms)')
        plt.axvline(x=onset_index * time_per_sample, color='green', linestyle='--', label='Movement Onset')
        plt.axvline(x=offset_index * time_per_sample, color='red', linestyle='--', label='End of Signal (Offset)')
        plt.xlabel("Time (ms)")
        plt.ylabel("Speed")
        plt.title(f"Speed Data with Movement Onset, Offset, and Movement Time: {movement_time:.2f} ms")
        plt.legend()
        plt.show()

    return movement_time, onset_index, offset_index

# Example usage:
# original_length = 2000  # Example original trial length in ms
# speed_data_resampled = np.concatenate([np.random.rand(12) * 0.01, np.random.rand(52) * 0.5])  # Simulated example data
# movement_time, onset_index, offset_index = calculate_movement_time(
#     speed_data_resampled, calculate_reaction_time_slope, original_length, slope_threshold=0.01, plot=True
# )



#%% max speed
# def calculate_max_speed_between_onset_offset(speed_data, detection_method, **kwargs):
#     """
#     Calculate the maximum speed between movement onset and offset based on advanced detection methods.
    
#     Parameters:
#     - speed_data: 1D numpy array containing the resampled speed values.
#     - detection_method: Function to detect movement onset and offset (Slope, CUSUM, or Moving Average).
#     - kwargs: Additional parameters for the detection method (thresholds, window size, etc.).
    
#     Returns:
#     - max_speed: The maximum speed within the detected movement phase.
#     - onset_index: The index of movement onset.
#     - offset_index: The index of movement offset.
#     """
#     # Detect onset and offset using the provided detection method
#     onset_index, offset_index = detection_method(speed_data, **kwargs)

#     # Ensure valid indices: If offset is not greater than onset, set offset to the end of the data
#     if offset_index <= onset_index:
#         offset_index = len(speed_data) - 1

#     # Extract the movement phase data
#     movement_phase_data = speed_data[onset_index:offset_index]

#     # Calculate the maximum speed within this phase
#     max_speed = np.max(movement_phase_data) if len(movement_phase_data) > 0 else 0.0

#     return max_speed, onset_index, offset_index



# # Parameters for each detection method
# slope_params = {'onset_threshold': 0.01, 'offset_threshold': 0.01}
# cusum_params = {'cusum_threshold': 0.02}
# ma_params = {'window_size': 5, 'ma_threshold': 0.02}

# # Calculate maximum speed and plot results for each method

# # Slope Method
# max_speed_slope, onset_slope, offset_slope = calculate_max_speed_between_onset_offset(
#     speed_data_resampled, calculate_movement_onset_offset_slope, **slope_params
# )

# # CUSUM Method
# max_speed_cusum, onset_cusum, offset_cusum = calculate_max_speed_between_onset_offset(
#     speed_data_resampled, calculate_movement_onset_offset_cusum, **cusum_params
# )

# # Moving Average Method
# max_speed_ma, onset_ma, offset_ma = calculate_max_speed_between_onset_offset(
#     speed_data_resampled, calculate_movement_onset_offset_moving_average, **ma_params
# )

# # Visualization
# plt.figure(figsize=(15, 12))

# # Slope Method
# plt.subplot(3, 1, 1)
# plt.plot(speed_data_resampled, label='Speed Data')
# plt.axvline(onset_slope, color='green', linestyle='--', label='Onset (Slope)')
# plt.axvline(offset_slope, color='red', linestyle='--', label='Offset (Slope)')
# plt.axhline(max_speed_slope, color='blue', linestyle='--', label=f'Max Speed ({max_speed_slope:.2f})')
# plt.fill_between(range(onset_slope, offset_slope), speed_data_resampled[onset_slope:offset_slope], alpha=0.3)
# plt.title(f'Slope Method - Maximum Speed: {max_speed_slope:.2f}')
# plt.xlabel('Sample Index')
# plt.ylabel('Speed')
# plt.legend()

# # CUSUM Method
# plt.subplot(3, 1, 2)
# plt.plot(speed_data_resampled, label='Speed Data')
# plt.axvline(onset_cusum, color='green', linestyle='--', label='Onset (CUSUM)')
# plt.axvline(offset_cusum, color='red', linestyle='--', label='Offset (CUSUM)')
# plt.axhline(max_speed_cusum, color='blue', linestyle='--', label=f'Max Speed ({max_speed_cusum:.2f})')
# plt.fill_between(range(onset_cusum, offset_cusum), speed_data_resampled[onset_cusum:offset_cusum], alpha=0.3)
# plt.title(f'CUSUM Method - Maximum Speed: {max_speed_cusum:.2f}')
# plt.xlabel('Sample Index')
# plt.ylabel('Speed')
# plt.legend()

# # Moving Average Method
# plt.subplot(3, 1, 3)
# plt.plot(speed_data_resampled, label='Speed Data')
# plt.axvline(onset_ma, color='green', linestyle='--', label='Onset (Moving Average)')
# plt.axvline(offset_ma, color='red', linestyle='--', label='Offset (Moving Average)')
# plt.axhline(max_speed_ma, color='blue', linestyle='--', label=f'Max Speed ({max_speed_ma:.2f})')
# plt.fill_between(range(onset_ma, offset_ma), speed_data_resampled[onset_ma:offset_ma], alpha=0.3)
# plt.title(f'Moving Average Method - Maximum Speed: {max_speed_ma:.2f}')
# plt.xlabel('Sample Index')
# plt.ylabel('Speed')
# plt.legend()

# plt.tight_layout()
# plt.show()



import numpy as np
import matplotlib.pyplot as plt

def calculate_max_speed_between_onset_offset(speed_data, detection_method, original_length, slope_threshold=0.01, plot=False):
    """
    Calculate the maximum speed between movement onset and offset based on advanced detection methods,
    with optional plotting for visualization.

    Parameters:
    - speed_data: 1D numpy array containing the resampled speed values.
    - detection_method: Function to detect movement onset (e.g., calculate_reaction_time_slope).
    - original_length: The total duration of the original trial in milliseconds.
    - slope_threshold: Threshold for the rate of change in speed for movement onset detection (default: 0.01).
    - plot: Boolean indicating whether to plot the detected movement onset, offset, and maximum speed (default: False).

    Returns:
    - max_speed: The maximum speed within the detected movement phase.
    - onset_index: The index of movement onset.
    - offset_index: The index of movement offset (end of signal).
    """
    # Detect movement onset after target-on (200 ms after the start)
    onset_index = int(detection_method(speed_data, slope_threshold=slope_threshold, original_length=original_length) * len(speed_data))
    
    # Set movement offset to the end of the signal
    offset_index = len(speed_data) - 1

    # Extract the movement phase data
    movement_phase_data = speed_data[onset_index:offset_index]

    # Calculate the maximum speed within this phase
    max_speed = np.max(movement_phase_data) if len(movement_phase_data) > 0 else 0.0

    # Plot if requested
    if plot:
        time_per_sample = original_length / len(speed_data)  # Time per sample in ms
        time_axis = np.arange(len(speed_data)) * time_per_sample  # Time axis for the entire signal
        movement_phase_time = time_axis[onset_index:offset_index]  # Time axis for the movement phase

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, speed_data, label='Speed Data', color='blue')
        plt.axvline(x=200, color='purple', linestyle='--', label='Target On (200 ms)')
        plt.axvline(x=onset_index * time_per_sample, color='green', linestyle='--', label='Movement Onset')
        plt.axvline(x=offset_index * time_per_sample, color='red', linestyle='--', label='End of Signal (Offset)')
        plt.scatter(movement_phase_time[np.argmax(movement_phase_data)], max_speed, color='red', label='Max Speed')
        plt.xlabel("Time (ms)")
        plt.ylabel("Speed")
        plt.title(f"Speed Data with Movement Onset, Offset, and Maximum Speed: {max_speed:.3f}")
        plt.legend()
        plt.show()

    return max_speed, onset_index, offset_index

# Example usage:
# original_length = 2000  # Example original trial length in ms
# speed_data_resampled = np.concatenate([np.random.rand(12) * 0.01, np.random.rand(52) * 0.5])  # Simulated example data
# max_speed, onset_index, offset_index = calculate_max_speed_between_onset_offset(
#     speed_data_resampled, calculate_reaction_time_slope, original_length, slope_threshold=0.01, plot=True
# )



#%%

