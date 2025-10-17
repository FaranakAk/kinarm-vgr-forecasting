# for Dharsan
import numpy as np
import csv
from aeon.datasets import write_to_tsfile
from sklearn.model_selection import train_test_split
import os

# Load control data
control_data = np.load('control_raw_VandP_Pcorrcted_subset.npz')
control_headers = control_data['all_header_CT']
control_Vabs = control_data['all_Vabs_CT']
control_Pabs = control_data['all_Pabs_CT']

# Load stroke data
stroke_data = np.load('stroke_raw_VandP_Pcorrected_subset.npz')
stroke_headers = stroke_data['all_header_ST']
stroke_Vabs = stroke_data['all_Vabs_ST']
stroke_Pabs = stroke_data['all_Pabs_ST']

# Function to extract unique subject-hand-datetime values and select 9 trials
def extract_trials(headers, Vabs, Pabs, label):
    unique_keys = {}
    for i, header in enumerate(headers):
        key = (header[0], header[1], header[-1].replace('.zip', ''))
        if key not in unique_keys:
            unique_keys[key] = {'Vabs': [], 'Pabs': [], 'headers': []}
        unique_keys[key]['Vabs'].append(Vabs[i])
        unique_keys[key]['Pabs'].append(Pabs[i])
        unique_keys[key]['headers'].append(header)
    
    selected_Vabs = []
    selected_Pabs = []
    selected_headers = []
    labels = []
    for key, data in unique_keys.items():
        if len(data['Vabs']) >= 9:
            selected_Vabs.append(np.vstack(data['Vabs'][:9]))
            selected_Pabs.append(np.vstack(data['Pabs'][:9]))
            selected_headers.append(data['headers'][:9])
            labels.append(label)
    
    return np.array(selected_Vabs), np.array(selected_Pabs), np.array(labels), selected_headers

# Extract trials for control and stroke groups
control_Vabs_data, control_Pabs_data, control_labels, control_headers = extract_trials(control_headers, control_Vabs, control_Pabs, 0)
stroke_Vabs_data, stroke_Pabs_data, stroke_labels, stroke_headers = extract_trials(stroke_headers, stroke_Vabs, stroke_Pabs, 1)

# Combine data
Vabs_data = np.concatenate([control_Vabs_data, stroke_Vabs_data])
Pabs_data = np.concatenate([control_Pabs_data, stroke_Pabs_data])
labels = np.concatenate([control_labels, stroke_labels])
headers = control_headers + stroke_headers

# Split into train and test sets
Vabs_train_data, Vabs_test_data, Vabs_train_labels, Vabs_test_labels, Vabs_train_headers, Vabs_test_headers = train_test_split(
    Vabs_data, labels, headers, test_size=0.2, random_state=42)

Pabs_train_data, Pabs_test_data, Pabs_train_labels, Pabs_test_labels, Pabs_train_headers, Pabs_test_headers = train_test_split(
    Pabs_data, labels, headers, test_size=0.2, random_state=42)

# Function to save data to .ts file
def save_to_tsfile(data, labels, filename):
    with open(filename, 'w') as f:
        f.write("@problemName VGR\n")
        f.write("@timestamps false\n")
        f.write("@missing false\n")
        f.write("@univariate true\n")
        f.write("@equalLength true\n")
        f.write("@classLabel true 0 1\n")
        f.write("@data\n")
        for i in range(len(data)):
            series = data[i]
            label = labels[i]
            for row in series:
                f.write(f"{','.join(map(str, row))}:{label}\n")
            f.write("\n")

# Function to save headers to a .csv file
def save_headers(headers, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['SubjectID', 'Hand', 'Task', 'DateTime'])
        for header_group in headers:
            for header in header_group:
                writer.writerow(header)

# Specify the directory where you have write permissions
output_dir = 'dataset/newdata'
os.makedirs(output_dir, exist_ok=True)

# Save Vabs and Pabs data to .ts files
save_to_tsfile(Vabs_train_data, Vabs_train_labels, os.path.join(output_dir, 'newdata_TRAIN.ts'))
save_to_tsfile(Vabs_test_data, Vabs_test_labels, os.path.join(output_dir, 'newdata_TEST.ts'))
save_to_tsfile(Pabs_train_data, Pabs_train_labels, os.path.join(output_dir, 'Pabs_train.ts'))
save_to_tsfile(Pabs_test_data, Pabs_test_labels, os.path.join(output_dir, 'Pabs_test.ts'))

# Save headers to .csv files
save_headers(Vabs_train_headers, os.path.join(output_dir, 'Vabs_train_headers.csv'))
save_headers(Vabs_test_headers, os.path.join(output_dir, 'Vabs_test_headers.csv'))
save_headers(Pabs_train_headers, os.path.join(output_dir, 'Pabs_train_headers.csv'))
save_headers(Pabs_test_headers, os.path.join(output_dir, 'Pabs_test_headers.csv'))

#%% for Nooshin - 1

import numpy as np
from scipy.signal import resample
import os
# os.chdir("D:/OneDrive - Queen's University/UNITS/weak")
os.chdir("D:/OneDrive - Queen's University/MOMENT/moment-main")
import h5py

# Load control data from HDF5
with h5py.File('new_control_raw_VandP_Pcorrected_cropped_3.h5', 'r') as f:
    control_headers = [f['all_header_CT'][str(i)][:] for i in range(len(f['all_header_CT']))]
    control_Feats = [f['all_features_CT'][str(i)][:] for i in range(len(f['all_features_CT']))]
    control_Vabs = [f['all_Vabs_CT'][str(i)][:] for i in range(len(f['all_Vabs_CT']))]
    control_Pabs = [f['all_Pabs_CT'][str(i)][:] for i in range(len(f['all_Pabs_CT']))]

# Load stroke data from HDF5
with h5py.File('new_stroke_raw_VandP_Pcorrected_cropped_3.h5', 'r') as f:
    stroke_headers = [f['all_header_ST'][str(i)][:] for i in range(len(f['all_header_ST']))]
    stroke_Feats = [f['all_features_ST'][str(i)][:] for i in range(len(f['all_features_ST']))]
    stroke_Vabs = [f['all_Vabs_ST'][str(i)][:] for i in range(len(f['all_Vabs_ST']))]
    stroke_Pabs = [f['all_Pabs_ST'][str(i)][:] for i in range(len(f['all_Pabs_ST']))]

# Function to decode headers
def decode_headers(headers):
    decoded_headers = []
    for header in headers:
        decoded_header = [field.decode('utf-8') if isinstance(field, bytes) else field for field in header]
        decoded_headers.append(decoded_header)
    return decoded_headers

# Decode headers
control_headers = decode_headers(control_headers)
stroke_headers = decode_headers(stroke_headers)

# Function to extract unique subject-hand-datetime values and select n_trials trials
def extract_trials_0(headers, Vabs, Pabs, Feats, label, num_samples=256, n_trials=9):
    unique_keys = {}
    for i, header in enumerate(headers):
        key = (header[0], header[1], header[-1].replace('.zip', ''))  # no need to decode as it's already handled
        if key not in unique_keys:
            unique_keys[key] = {'Vabs': [], 'Pabs': [], 'Feats': [], 'headers': []}
        unique_keys[key]['Vabs'].append(Vabs[i])
        unique_keys[key]['Pabs'].append(Pabs[i])
        unique_keys[key]['Feats'].append(Feats[i])
        unique_keys[key]['headers'].append(header)
    
    selected_Vabs = []
    selected_Pabs = []
    selected_Feats = []
    selected_headers = []
    labels = []
    for key, data in unique_keys.items():
        if len(data['Vabs']) >= n_trials:
            # Select and downsample the first n_trials trials
            selected_Vabs.append(np.concatenate([resample(v, num_samples) for v in data['Vabs'][:n_trials]]))
            selected_Pabs.append(np.concatenate([resample(p, num_samples) for p in data['Pabs'][:n_trials]]))
            selected_Feats.append(data['Feats'][0])  # Only save features from the first selected trial
            selected_headers.append(data['headers'][0])  # Only save headers from the first selected trial
            labels.append(label)
    
    return (np.array(selected_Vabs), np.array(selected_Pabs), 
            np.array(selected_Feats), np.array(labels), selected_headers)





def extract_trials(headers, Vabs, Pabs, Feats, label, num_samples=256, n_trials=9):
    """
    Extracts trials for unique subject-hand-session combinations.
    
    Parameters:
    - headers: List of metadata for each trial. Each header has [subject_id, direction, session, ...].
    - Vabs: List of velocity time-series for each trial.
    - Pabs: List of position time-series for each trial.
    - Feats: List of features for each trial.
    - label: Label assigned to each subject/session (e.g., 0 for control, 1 for stroke).
    - num_samples: Number of samples to downsample each trial to.
    - n_trials: Number of trials to select for each unique subject-hand-session.
    
    Returns:
    - selected_Vabs: Concatenated Vabs data for selected trials.
    - selected_Pabs: Concatenated Pabs data for selected trials.
    - selected_Feats: Features of the first selected trial.
    - labels: List of labels for each unique subject-hand-session.
    - selected_headers: List of headers for each selected trial.
    - selected_directions: List of movement directions for each selected trial.
    - selected_indices_all: List of indices of the selected trials relative to the total trials for each subject/session.
    """
    
    # Direction mapping if header[-3] is 1
    direction_map = {'4': '8', '3': '7', '2': '6', '9': '5', '8': '4', '7': '3', '6': '2', '5': '9'}
    
    # Group trials by unique keys (subject, session)
    unique_keys = {}
    for i, header in enumerate(headers):
        key = (header[0], header[2], header[-1].replace('.zip', ''))  # Unique key (subject_id, session)
        
        # Check if header[-3] is 1, and adjust the direction accordingly
        direction = header[1]
        if header[-3] == '1' and direction in direction_map:
            direction = direction_map[direction]
        
        if key not in unique_keys:
            unique_keys[key] = {'Vabs': [], 'Pabs': [], 'Feats': [], 'headers': [], 'directions': []}
        
        unique_keys[key]['Vabs'].append(Vabs[i])
        unique_keys[key]['Pabs'].append(Pabs[i])
        unique_keys[key]['Feats'].append(Feats[i])
        unique_keys[key]['headers'].append(header)
        unique_keys[key]['directions'].append(direction)
    
    selected_Vabs = []
    selected_Pabs = []
    selected_Feats = []
    selected_headers = []
    selected_indices_all = []  # Store the indices of the selected trials for each subject/session
    selected_directions_all = []  # Store the directions of the selected trials for each subject/session
    labels = []
    
    for key, data in unique_keys.items():
        directions = np.array(data['directions'])
        unique_directions = np.unique(directions)
        
        if len(data['Vabs']) >= n_trials:
            selected_indices = []
            
            if len(unique_directions) >= n_trials: 
                # If there are more unique directions than required trials, pick one trial from each of n unique directions
                selected_directions = np.random.choice(unique_directions, size=n_trials, replace=False)
                for direction in selected_directions:
                    direction_indices = np.where(directions == direction)[0]
                    chosen_index = np.random.choice(direction_indices)
                    selected_indices.append(chosen_index)
            else:
                # If there are fewer unique directions, divide n_trials among the available directions
                num_per_direction = n_trials // len(unique_directions)
                extra_trials = n_trials % len(unique_directions)
                
                for direction in unique_directions:
                    direction_indices = np.where(directions == direction)[0]
                    if len(direction_indices) >= num_per_direction:
                        chosen_indices = np.random.choice(direction_indices, size=num_per_direction, replace=False)
                    else:
                        # If there aren't enough trials for this direction, just select all of them
                        chosen_indices = direction_indices
                    selected_indices.extend(chosen_indices)
                
                # If n_trials is not a perfect multiple of the number of unique directions, select the remaining trials randomly
                if extra_trials > 0:
                    remaining_indices = list(set(range(len(directions))) - set(selected_indices))
                    extra_selected = np.random.choice(remaining_indices, size=extra_trials, replace=False)
                    selected_indices.extend(extra_selected)
            
            # Sort indices to ensure consistent ordering
            selected_indices = sorted(set(selected_indices))
            
            # Track selected indices for this subject/session
            selected_indices_all.append(selected_indices)
            
            # Select and downsample the first n_trials trials
            selected_Vabs.append(np.concatenate([resample(data['Vabs'][i], num_samples) for i in selected_indices]))
            selected_Pabs.append(np.concatenate([resample(data['Pabs'][i], num_samples) for i in selected_indices]))
            selected_Feats.append(data['Feats'][selected_indices[0]])  # Use features from the first selected trial
            selected_headers.append(data['headers'][selected_indices[0]])  # Use headers from the first selected trial
            
            # Extract the directions of the selected trials
            selected_directions = [data['directions'][i] for i in selected_indices]
            selected_directions_all.append(selected_directions)
            
            labels.append(label)
    
    return (
        np.array(selected_Vabs), 
        np.array(selected_Pabs), 
        np.array(selected_Feats), 
        np.array(labels), 
        selected_headers,
        
        selected_indices_all,  # Return the selected indices for each subject/session
        selected_directions_all,  # Return the directions of the selected trials
    )



# Extract trials for control and stroke groups
control_Vabs_data, control_Pabs_data, control_Feats_data, control_labels, control_headers, control_selected_indices, control_directions = extract_trials(control_headers, control_Vabs, control_Pabs, control_Feats, 0)
stroke_Vabs_data, stroke_Pabs_data, stroke_Feats_data, stroke_labels, stroke_headers, stroke_selected_indices, stroke_directions = extract_trials(stroke_headers, stroke_Vabs, stroke_Pabs, stroke_Feats, 1)

# Function to save variable length data in HDF5
def save_variable_length_data(h5file, dataset_name, data):
    grp = h5file.create_group(dataset_name)
    for i, item in enumerate(data):
        grp.create_dataset(str(i), data=item, compression="gzip")

# Save the selected data to new HDF5 files for control
with h5py.File('selected_control_raw_VandP_Pcorrected_cropped_3.h5', 'w') as f:
    save_variable_length_data(f, 'Vabs', control_Vabs_data)
    save_variable_length_data(f, 'Pabs', control_Pabs_data)
    save_variable_length_data(f, 'Feats', control_Feats_data)
    save_variable_length_data(f, 'headers', control_headers)
    save_variable_length_data(f, 'selected_indices', control_selected_indices)
    save_variable_length_data(f, 'directions', control_directions)

# Save the selected data to new HDF5 files for stroke
with h5py.File('selected_stroke_raw_VandP_Pcorrected_cropped_3.h5', 'w') as f:
    save_variable_length_data(f, 'Vabs', stroke_Vabs_data)
    save_variable_length_data(f, 'Pabs', stroke_Pabs_data)
    save_variable_length_data(f, 'Feats', stroke_Feats_data)
    save_variable_length_data(f, 'headers', stroke_headers)
    save_variable_length_data(f, 'selected_indices', stroke_selected_indices)
    save_variable_length_data(f, 'directions', stroke_directions)
    



#%% for Nooshin - 2

import numpy as np
from scipy.signal import resample
import os
os.chdir("D:/OneDrive - Queen's University/MOMENT/moment-main")
import h5py

##############################################################################
# 1) Parameters and which group(s) to process
##############################################################################

GROUP_TO_PROCESS = 'both'  # options: 'control', 'stroke', 'both'
NUM_SAMPLES = 256
AUGMENT_MODE = 'random' #options: 'distinct', 'random

CONTROL_INPUT_FILENAME = 'new_control_raw_VandP_Pcorrected_cropped_Mar_2025_adv.h5'
STROKE_INPUT_FILENAME = 'new_stroke_raw_VandP_Pcorrected_cropped_Mar_2025_adv.h5'

CONTROL_OUTPUT_FILENAME = 'selected_control_raw_VandP_Pcorrected_cropped_Mar_2025_adv.h5'
STROKE_OUTPUT_FILENAME = 'selected_stroke_raw_VandP_Pcorrected_cropped_Mar_2025_adv.h5'


##############################################################################
# 2) Utility functions
##############################################################################

def decode_headers(headers):
    """
    Decode any byte-strings in the headers to UTF-8 strings.
    """
    decoded_headers = []
    for header in headers:
        decoded_header = [
            field.decode('utf-8') if isinstance(field, bytes) else field
            for field in header
        ]
        decoded_headers.append(decoded_header)
    return decoded_headers

def save_variable_length_data(h5file, dataset_name, data):
    """
    Save a list (or array) of variable-length items into an HDF5 file.
    Each 'item' becomes its own HDF5 dataset inside a group `dataset_name`.
    """
    grp = h5file.create_group(dataset_name)
    for i, item in enumerate(data):
        grp.create_dataset(str(i), data=item, compression="gzip")

##############################################################################
# 3) Augmented extraction logic (no "labels" or "selected_indices" in output)
##############################################################################

def extract_trials_augmented(headers, Vabs, Pabs, Feats, label, 
                             num_samples=256, aug_mode="random"):
    """
    For each unique subject-hand-session combination, create data samples.
    
    Two modes are available (set via aug_mode):
    
    - "distinct": 
      For subjects with 8 directions:
        * Base 8 = first trial from each of the 8 directions.
        * Then produce 8 data samples, each adding the second trial from one direction
          as the 9th trial (one per direction).
      For subjects with 4 directions:
        * Base 8 = first TWO trials from each of the 4 directions (2×4=8).
        * Then produce 8 data samples, each adding one additional trial from one direction.
    
    - "random":
      For subjects with 8 directions:
        * Base 8 = first trial from each of the 8 directions.
        * Then produce ONE data sample by randomly choosing one of the 8 candidate second trials.
      For subjects with 4 directions:
        * Base 8 = first TWO trials from each of the 4 directions.
        * Then produce ONE data sample by randomly choosing one of the extra candidate trials.
    
    Returns:
      - all_selected_Vabs
      - all_selected_Pabs
      - all_selected_Feats
      - all_selected_headers
      - all_selected_directions   (list of direction arrays, each length 9)
      - all_selected_lengths      (list of arrays of original lengths for the 9 trials)
    """
    
    direction_map = {
        '4': '8', '3': '7', '2': '6', '9': '5', 
        '8': '4', '7': '3', '6': '2', '5': '9'
    }
    
    # Group trials by (subject_id, session, date)
    unique_keys = {}
    for i, header in enumerate(headers):
        key = (header[0], header[2], header[-1].replace('.zip', ''))
        
        # Mirror direction if header[-3] == '1'
        direction = header[1]
        if header[-3] == '1' and direction in direction_map:
            direction = direction_map[direction]
        
        if key not in unique_keys:
            unique_keys[key] = {
                'Vabs': [], 'Pabs': [], 'Feats': [],
                'headers': [], 'directions': []
            }
        unique_keys[key]['Vabs'].append(Vabs[i])
        unique_keys[key]['Pabs'].append(Pabs[i])
        unique_keys[key]['Feats'].append(Feats[i])
        unique_keys[key]['headers'].append(header)
        unique_keys[key]['directions'].append(direction)
    
    # Final outputs
    all_selected_Vabs = []
    all_selected_Pabs = []
    all_selected_Feats = []
    all_selected_headers = []
    all_selected_directions = []  # store an array of 9 directions per data sample
    all_selected_lengths = []      # new: store original lengths of the 9 trials
    
    for key, data in unique_keys.items():
        directions = np.array(data['directions'])
        unique_dirs = np.unique(directions)
        num_dirs = len(unique_dirs)
        
        # We only handle exactly 4- or 8-direction subjects
        if num_dirs not in [4, 8]:
            continue
        
        # Build a dictionary: direction -> list of trial indices
        dir2indices = {}
        for idx, d in enumerate(directions):
            dir2indices.setdefault(d, []).append(idx)
        # Sort each list so "first trial" is the earliest
        for d in dir2indices:
            dir2indices[d].sort()
        
        if num_dirs == 8:
            # Need >=2 trials in each of 8 directions
            if not all(len(dir2indices[d]) >= 2 for d in unique_dirs):
                continue
            
            # Base 8 = 1st trial from each direction
            base_indices = []
            base_directions = []
            for d in unique_dirs:
                base_indices.append(dir2indices[d][0])
                base_directions.append(d)
            
            # We have 8 'second' trials, one from each direction
            second_indices = []
            second_directions = []
            for d in unique_dirs:
                second_indices.append(dir2indices[d][1])
                second_directions.append(d)
            
            if aug_mode == "distinct":
                # Produce 8 augmented samples
                for i_dir in range(8):
                    # The 9th trial is the second trial from direction i_dir
                    idx_9th = second_indices[i_dir]
                    dir_9th = second_directions[i_dir]
                    
                    selected_indices = base_indices + [idx_9th]
                    selected_dirs = base_directions + [dir_9th]
                    
                    # Downsample and concatenate velocity
                    v_concat = np.concatenate([
                        resample(data['Vabs'][i], num_samples) 
                        for i in selected_indices
                    ])
                    # Downsample and concatenate position
                    p_concat = np.concatenate([
                        resample(data['Pabs'][i], num_samples) 
                        for i in selected_indices
                    ])
                    
                    # Original lengths array for these 9 trials
                    orig_lens = np.array([len(data['Vabs'][i]) for i in selected_indices])
                    
                    # Use features and header from the first base trial
                    feat_selected = data['Feats'][base_indices[0]]
                    header_selected = data['headers'][base_indices[0]]
                    
                    all_selected_Vabs.append(v_concat)
                    all_selected_Pabs.append(p_concat)
                    all_selected_Feats.append(feat_selected)
                    all_selected_headers.append(header_selected)
                    all_selected_directions.append(selected_dirs)
                    all_selected_lengths.append(orig_lens)
                    
            elif aug_mode == "random":
                # Produce one augmented sample by randomly choosing one candidate
                i_dir = np.random.randint(0, 8)
                idx_9th = second_indices[i_dir]
                dir_9th = second_directions[i_dir]
                selected_indices = base_indices + [idx_9th]
                selected_dirs = base_directions + [dir_9th]
                
                v_concat = np.concatenate([
                    resample(data['Vabs'][i], num_samples) 
                    for i in selected_indices
                ])
                p_concat = np.concatenate([
                    resample(data['Pabs'][i], num_samples) 
                    for i in selected_indices
                ])
                orig_lens = np.array([len(data['Vabs'][i]) for i in selected_indices])
                
                feat_selected = data['Feats'][base_indices[0]]
                header_selected = data['headers'][base_indices[0]]
                
                all_selected_Vabs.append(v_concat)
                all_selected_Pabs.append(p_concat)
                all_selected_Feats.append(feat_selected)
                all_selected_headers.append(header_selected)
                all_selected_directions.append(selected_dirs)
                all_selected_lengths.append(orig_lens)
                
        elif num_dirs == 4:
            # Need >=4 trials per each direction
            if not all(len(dir2indices[d]) >= 4 for d in unique_dirs):
                continue
            
            # Base 8 => first 2 trials from each direction (2×4=8)
            base_indices = []
            base_directions = []
            for d in unique_dirs:
                first_two = dir2indices[d][0:2]
                base_indices.extend(first_two)
                base_directions.extend([d, d])  # 2 trials => same direction repeated
            
            # Next 2 trials from each direction => total 2×4=8 for the 9th slot
            extra_indices = []
            extra_directions = []
            for d in unique_dirs:
                # the next two after the first two => [2:4]
                next_two = dir2indices[d][2:4]
                extra_indices.extend(next_two)
                extra_directions.extend([d]*len(next_two))
                
            if aug_mode == "distinct":
            
                # Should be exactly 8 extra trial indices
                for i_idx in range(len(extra_indices)):
                    idx_9th = extra_indices[i_idx]
                    dir_9th = extra_directions[i_idx]
                    
                    selected_indices = base_indices + [idx_9th]
                    selected_dirs = base_directions + [dir_9th]
                    
                    # Downsample and concatenate velocity
                    v_concat = np.concatenate([
                        resample(data['Vabs'][i], num_samples) 
                        for i in selected_indices
                    ])
                    # Downsample and concatenate position
                    p_concat = np.concatenate([
                        resample(data['Pabs'][i], num_samples) 
                        for i in selected_indices
                    ])
                    
                    # Original lengths array for these 9 trials
                    orig_lens = np.array([len(data['Vabs'][i]) for i in selected_indices])
                    
                    # Use features, header from the first base trial
                    feat_selected = data['Feats'][base_indices[0]]
                    header_selected = data['headers'][base_indices[0]]
                    
                    all_selected_Vabs.append(v_concat)
                    all_selected_Pabs.append(p_concat)
                    all_selected_Feats.append(feat_selected)
                    all_selected_headers.append(header_selected)
                    all_selected_directions.append(selected_dirs)
                    all_selected_lengths.append(orig_lens)
    
            elif aug_mode == "random":
                # Produce one augmented sample by randomly choosing one extra trial
                i_idx = np.random.randint(0, len(extra_indices))
                idx_9th = extra_indices[i_idx]
                dir_9th = extra_directions[i_idx]
                selected_indices = base_indices + [idx_9th]
                selected_dirs = base_directions + [dir_9th]
                
                v_concat = np.concatenate([
                    resample(data['Vabs'][i], num_samples) 
                    for i in selected_indices
                ])
                p_concat = np.concatenate([
                    resample(data['Pabs'][i], num_samples) 
                    for i in selected_indices
                ])
                orig_lens = np.array([len(data['Vabs'][i]) for i in selected_indices])
                
                feat_selected = data['Feats'][base_indices[0]]
                header_selected = data['headers'][base_indices[0]]
                
                all_selected_Vabs.append(v_concat)
                all_selected_Pabs.append(p_concat)
                all_selected_Feats.append(feat_selected)
                all_selected_headers.append(header_selected)
                all_selected_directions.append(selected_dirs)
                all_selected_lengths.append(orig_lens)
    # Convert directions to arrays of byte-strings so we can store them easily
    # each element is a list of (9) direction strings
    # We'll store them as a "variable-length" approach with save_variable_length_data
    # Example: convert each direction list to np.array(dtype='S2')
    final_dirs = []
    for dirs_list in all_selected_directions:
        arr = np.array(dirs_list, dtype='S2')  # if single-digit or 2-digit strings
        final_dirs.append(arr)
    
    return (
        np.array(all_selected_Vabs),
        np.array(all_selected_Pabs),
        np.array(all_selected_Feats),
        all_selected_headers,
        final_dirs,     # list of (9,) arrays for directions
        all_selected_lengths         # list of (9,) arrays for original lengths
        )  
    


##############################################################################
# 4) Main flow: load, extract, save  (same keys, minus 'selected_indices')
##############################################################################


if GROUP_TO_PROCESS in ['both', 'control']:
    # Load control data
    with h5py.File(CONTROL_INPUT_FILENAME, 'r') as f:
        control_headers = [f['all_header_CT'][str(i)][:] 
                           for i in range(len(f['all_header_CT']))]
        control_Feats = [f['all_features_CT'][str(i)][:] 
                         for i in range(len(f['all_features_CT']))]
        control_Vabs = [f['all_Vabs_CT'][str(i)][:] 
                        for i in range(len(f['all_Vabs_CT']))]
        control_Pabs = [f['all_Pabs_CT'][str(i)][:] 
                        for i in range(len(f['all_Pabs_CT']))]
    
    # Decode headers
    control_headers = decode_headers(control_headers)
    
    # Extract with augmented logic
    c_Vabs_data, c_Pabs_data, c_Feats_data, c_headers_data, c_directions_data, c_lengths_data = (
        extract_trials_augmented(
            control_headers, control_Vabs, control_Pabs, control_Feats,
            label=0,  # 'control' label, though we'll NOT save it
            num_samples=NUM_SAMPLES,
            aug_mode="random"
        )
    )
    
    # Save (same keys, minus 'selected_indices')
    with h5py.File(CONTROL_OUTPUT_FILENAME, 'w') as f:
        save_variable_length_data(f, 'Vabs', c_Vabs_data)
        save_variable_length_data(f, 'Pabs', c_Pabs_data)
        save_variable_length_data(f, 'Feats', c_Feats_data)
        save_variable_length_data(f, 'headers', c_headers_data)
        save_variable_length_data(f, 'directions', c_directions_data)
        save_variable_length_data(f, 'orig_lengths', c_lengths_data)

if GROUP_TO_PROCESS in ['both', 'stroke']:
    # Load stroke data
    with h5py.File(STROKE_INPUT_FILENAME, 'r') as f:
        stroke_headers = [f['all_header_ST'][str(i)][:] 
                          for i in range(len(f['all_header_ST']))]
        stroke_Feats = [f['all_features_ST'][str(i)][:] 
                        for i in range(len(f['all_features_ST']))]
        stroke_Vabs = [f['all_Vabs_ST'][str(i)][:] 
                       for i in range(len(f['all_Vabs_ST']))]
        stroke_Pabs = [f['all_Pabs_ST'][str(i)][:] 
                       for i in range(len(f['all_Pabs_ST']))]
    
    # Decode headers
    stroke_headers = decode_headers(stroke_headers)
    
    # Extract with augmented logic
    s_Vabs_data, s_Pabs_data, s_Feats_data, s_headers_data, s_directions_data, s_lengths_data = (
        extract_trials_augmented(
            stroke_headers, stroke_Vabs, stroke_Pabs, stroke_Feats,
            label=1,  # 'stroke' label, though we'll NOT save it
            num_samples=NUM_SAMPLES,
            aug_mode="random"
        )
    )
    
    # Save (same keys, minus 'selected_indices')
    with h5py.File(STROKE_OUTPUT_FILENAME, 'w') as f:
        save_variable_length_data(f, 'Vabs', s_Vabs_data)
        save_variable_length_data(f, 'Pabs', s_Pabs_data)
        save_variable_length_data(f, 'Feats', s_Feats_data)
        save_variable_length_data(f, 'headers', s_headers_data)
        save_variable_length_data(f, 'directions', s_directions_data)
        save_variable_length_data(f, 'orig_lengths', s_lengths_data)


#%% for Nooshin 2 - 2

import numpy as np
from scipy.signal import resample
import os
os.chdir("D:/OneDrive - Queen's University/MOMENT/moment-main")
import h5py

##############################################################################
# 1) Parameters and which group(s) to process
##############################################################################

GROUP_TO_PROCESS = 'both'  # options: 'control', 'stroke', 'both'
NUM_SAMPLES = 256
AUGMENT_DATA = False  # Set to False to use only the base 8-trial context per subject

CONTROL_INPUT_FILENAME = 'new_control_raw_VandP_Pcorrected_cropped_Mar_2025_adv.h5'
STROKE_INPUT_FILENAME = 'new_stroke_raw_VandP_Pcorrected_cropped_Mar_2025_adv.h5'

CONTROL_OUTPUT_FILENAME = 'selected_control_raw_VandP_Pcorrected_cropped_Mar_2025_adv.h5'
STROKE_OUTPUT_FILENAME = 'selected_stroke_raw_VandP_Pcorrected_cropped_Mar_2025_adv.h5'

##############################################################################
# 2) Utility functions
##############################################################################

def decode_headers(headers):
    """
    Decode any byte-strings in the headers to UTF-8 strings.
    """
    decoded_headers = []
    for header in headers:
        decoded_header = [
            field.decode('utf-8') if isinstance(field, bytes) else field
            for field in header
        ]
        decoded_headers.append(decoded_header)
    return decoded_headers

def save_variable_length_data(h5file, dataset_name, data):
    """
    Save a list (or array) of variable-length items into an HDF5 file.
    Each 'item' becomes its own HDF5 dataset inside a group `dataset_name`.
    """
    grp = h5file.create_group(dataset_name)
    for i, item in enumerate(data):
        grp.create_dataset(str(i), data=item, compression="gzip")

##############################################################################
# 3) Extraction logic with optional augmentation
##############################################################################

def extract_trials(headers, Vabs, Pabs, Feats, label, num_samples=256, augment=True):
    """
    For each unique subject-hand-session combination, create data samples.
    
    When augment=True:
      - For 8-direction subjects:
         * Base 8 = first trial from each of the 8 directions.
         * Then produce 8 samples, each with one additional (second) trial 
           from one direction added as the 9th trial.
      - For 4-direction subjects:
         * Base 8 = first TWO trials from each direction.
         * Then produce 8 samples, each adding one extra trial (3rd or 4th) 
           from one direction.
           
    When augment=False:
      - Only the base 8 trials are selected for each unique subject-key combination.
    
    Returns:
      - all_selected_Vabs
      - all_selected_Pabs
      - all_selected_Feats
      - all_selected_headers
      - all_selected_directions   (each is a list of directions for the trial set)
      - all_selected_lengths      (original lengths of the trials)
    """
    
    direction_map = {
        '4': '8', '3': '7', '2': '6', '9': '5', 
        '8': '4', '7': '3', '6': '2', '5': '9'
    }
    
    # Group trials by (subject_id, session, date)
    unique_keys = {}
    for i, header in enumerate(headers):
        key = (header[0], header[2], header[-1].replace('.zip', ''))
        
        # Mirror direction if header[-3] == '1'
        direction = header[1]
        if header[-3] == '1' and direction in direction_map:
            direction = direction_map[direction]
        
        if key not in unique_keys:
            unique_keys[key] = {
                'Vabs': [], 'Pabs': [], 'Feats': [],
                'headers': [], 'directions': []
            }
        unique_keys[key]['Vabs'].append(Vabs[i])
        unique_keys[key]['Pabs'].append(Pabs[i])
        unique_keys[key]['Feats'].append(Feats[i])
        unique_keys[key]['headers'].append(header)
        unique_keys[key]['directions'].append(direction)
    
    # Final outputs
    all_selected_Vabs = []
    all_selected_Pabs = []
    all_selected_Feats = []
    all_selected_headers = []
    all_selected_directions = []  # List of direction arrays per sample
    all_selected_lengths = []     # List of original lengths for the trials
    
    for key, data in unique_keys.items():
        directions = np.array(data['directions'])
        unique_dirs = np.unique(directions)
        num_dirs = len(unique_dirs)
        
        # Only process subjects with exactly 4 or 8 unique directions
        if num_dirs not in [4, 8]:
            continue
        
        # Build dictionary: direction -> list of trial indices
        dir2indices = {}
        for idx, d in enumerate(directions):
            dir2indices.setdefault(d, []).append(idx)
        # Ensure trials are in order (first trial is the earliest)
        for d in dir2indices:
            dir2indices[d].sort()
        
        if num_dirs == 8:
            # For augmentation, need at least 2 trials per direction.
            # For non-augmentation, need at least 1 trial per direction.
            if augment:
                if not all(len(dir2indices[d]) >= 2 for d in unique_dirs):
                    continue
            else:
                if not all(len(dir2indices[d]) >= 1 for d in unique_dirs):
                    continue
            
            # Base 8 = first trial from each direction
            base_indices = []
            base_directions = []
            for d in unique_dirs:
                base_indices.append(dir2indices[d][0])
                base_directions.append(d)
            
            if augment:
                # Gather second trials from each direction
                second_indices = []
                second_directions = []
                for d in unique_dirs:
                    second_indices.append(dir2indices[d][1])
                    second_directions.append(d)
                
                # Produce 8 augmented samples
                for i_dir in range(8):
                    # Use the second trial from one direction as the extra (9th) trial
                    idx_9th = second_indices[i_dir]
                    dir_9th = second_directions[i_dir]
                    
                    selected_indices = base_indices + [idx_9th]
                    selected_dirs = base_directions + [dir_9th]
                    
                    # Downsample and concatenate velocity
                    v_concat = np.concatenate([
                        resample(data['Vabs'][i], num_samples) 
                        for i in selected_indices
                    ])
                    # Downsample and concatenate position
                    p_concat = np.concatenate([
                        resample(data['Pabs'][i], num_samples) 
                        for i in selected_indices
                    ])
                    
                    # Record original lengths of each trial
                    orig_lens = np.array([len(data['Vabs'][i]) for i in selected_indices])
                    
                    # Use features and header from the first base trial
                    feat_selected = data['Feats'][base_indices[0]]
                    header_selected = data['headers'][base_indices[0]]
                    
                    all_selected_Vabs.append(v_concat)
                    all_selected_Pabs.append(p_concat)
                    all_selected_Feats.append(feat_selected)
                    all_selected_headers.append(header_selected)
                    all_selected_directions.append(selected_dirs)
                    all_selected_lengths.append(orig_lens)
            else:
                # No augmentation: only use the base 8 trials
                selected_indices = base_indices
                selected_dirs = base_directions
                v_concat = np.concatenate([
                    resample(data['Vabs'][i], num_samples)
                    for i in selected_indices
                ])
                p_concat = np.concatenate([
                    resample(data['Pabs'][i], num_samples)
                    for i in selected_indices
                ])
                orig_lens = np.array([len(data['Vabs'][i]) for i in selected_indices])
                feat_selected = data['Feats'][base_indices[0]]
                header_selected = data['headers'][base_indices[0]]
                
                all_selected_Vabs.append(v_concat)
                all_selected_Pabs.append(p_concat)
                all_selected_Feats.append(feat_selected)
                all_selected_headers.append(header_selected)
                all_selected_directions.append(selected_dirs)
                all_selected_lengths.append(orig_lens)
                
        elif num_dirs == 4:
            # For 4-direction subjects:
            # For augmentation, need at least 4 trials per direction.
            # For non-augmentation, need at least 2 trials per direction.
            if augment:
                if not all(len(dir2indices[d]) >= 4 for d in unique_dirs):
                    continue
            else:
                if not all(len(dir2indices[d]) >= 2 for d in unique_dirs):
                    continue
            
            # Base 8 = first 2 trials from each direction (2×4=8)
            base_indices = []
            base_directions = []
            for d in unique_dirs:
                first_two = dir2indices[d][0:2]
                base_indices.extend(first_two)
                base_directions.extend([d, d])
            
            if augment:
                # Next 2 trials from each direction for augmentation (total 8 options)
                extra_indices = []
                extra_directions = []
                for d in unique_dirs:
                    next_two = dir2indices[d][2:4]
                    extra_indices.extend(next_two)
                    extra_directions.extend([d] * len(next_two))
                
                # Produce 8 augmented samples
                for i_idx in range(len(extra_indices)):
                    idx_9th = extra_indices[i_idx]
                    dir_9th = extra_directions[i_idx]
                    
                    selected_indices = base_indices + [idx_9th]
                    selected_dirs = base_directions + [dir_9th]
                    
                    v_concat = np.concatenate([
                        resample(data['Vabs'][i], num_samples) 
                        for i in selected_indices
                    ])
                    p_concat = np.concatenate([
                        resample(data['Pabs'][i], num_samples) 
                        for i in selected_indices
                    ])
                    
                    orig_lens = np.array([len(data['Vabs'][i]) for i in selected_indices])
                    
                    feat_selected = data['Feats'][base_indices[0]]
                    header_selected = data['headers'][base_indices[0]]
                    
                    all_selected_Vabs.append(v_concat)
                    all_selected_Pabs.append(p_concat)
                    all_selected_Feats.append(feat_selected)
                    all_selected_headers.append(header_selected)
                    all_selected_directions.append(selected_dirs)
                    all_selected_lengths.append(orig_lens)
            else:
                # No augmentation: only use the base 8 trials (first 2 from each direction)
                selected_indices = base_indices
                selected_dirs = base_directions
                v_concat = np.concatenate([
                    resample(data['Vabs'][i], num_samples)
                    for i in selected_indices
                ])
                p_concat = np.concatenate([
                    resample(data['Pabs'][i], num_samples)
                    for i in selected_indices
                ])
                orig_lens = np.array([len(data['Vabs'][i]) for i in selected_indices])
                feat_selected = data['Feats'][base_indices[0]]
                header_selected = data['headers'][base_indices[0]]
                
                all_selected_Vabs.append(v_concat)
                all_selected_Pabs.append(p_concat)
                all_selected_Feats.append(feat_selected)
                all_selected_headers.append(header_selected)
                all_selected_directions.append(selected_dirs)
                all_selected_lengths.append(orig_lens)
    
    # Convert directions to arrays of byte-strings for storage
    final_dirs = []
    for dirs_list in all_selected_directions:
        arr = np.array(dirs_list, dtype='S2')
        final_dirs.append(arr)
    
    return (
        np.array(all_selected_Vabs),
        np.array(all_selected_Pabs),
        np.array(all_selected_Feats),
        all_selected_headers,
        final_dirs,     
        all_selected_lengths
    )

##############################################################################
# 4) Main flow: load, extract, save  (same keys, minus 'selected_indices')
##############################################################################

if GROUP_TO_PROCESS in ['both', 'control']:
    # Load control data
    with h5py.File(CONTROL_INPUT_FILENAME, 'r') as f:
        control_headers = [f['all_header_CT'][str(i)][:] 
                           for i in range(len(f['all_header_CT']))]
        control_Feats = [f['all_features_CT'][str(i)][:] 
                         for i in range(len(f['all_features_CT']))]
        control_Vabs = [f['all_Vabs_CT'][str(i)][:] 
                        for i in range(len(f['all_Vabs_CT']))]
        control_Pabs = [f['all_Pabs_CT'][str(i)][:] 
                        for i in range(len(f['all_Pabs_CT']))]
    
    # Decode headers
    control_headers = decode_headers(control_headers)
    
    # Extract trials (augmented or not, based on AUGMENT_DATA)
    c_Vabs_data, c_Pabs_data, c_Feats_data, c_headers_data, c_directions_data, c_lengths_data = (
        extract_trials(
            control_headers, control_Vabs, control_Pabs, control_Feats,
            label=0,  # 'control' label, not saved
            num_samples=NUM_SAMPLES,
            augment=AUGMENT_DATA
        )
    )
    
    # Save the processed control data
    with h5py.File(CONTROL_OUTPUT_FILENAME, 'w') as f:
        save_variable_length_data(f, 'Vabs', c_Vabs_data)
        save_variable_length_data(f, 'Pabs', c_Pabs_data)
        save_variable_length_data(f, 'Feats', c_Feats_data)
        save_variable_length_data(f, 'headers', c_headers_data)
        save_variable_length_data(f, 'directions', c_directions_data)
        save_variable_length_data(f, 'orig_lengths', c_lengths_data)

if GROUP_TO_PROCESS in ['both', 'stroke']:
    # Load stroke data
    with h5py.File(STROKE_INPUT_FILENAME, 'r') as f:
        stroke_headers = [f['all_header_ST'][str(i)][:] 
                          for i in range(len(f['all_header_ST']))]
        stroke_Feats = [f['all_features_ST'][str(i)][:] 
                        for i in range(len(f['all_features_ST']))]
        stroke_Vabs = [f['all_Vabs_ST'][str(i)][:] 
                       for i in range(len(f['all_Vabs_ST']))]
        stroke_Pabs = [f['all_Pabs_ST'][str(i)][:] 
                       for i in range(len(f['all_Pabs_ST']))]
    
    # Decode headers
    stroke_headers = decode_headers(stroke_headers)
    
    # Extract trials (augmented or not)
    s_Vabs_data, s_Pabs_data, s_Feats_data, s_headers_data, s_directions_data, s_lengths_data = (
        extract_trials(
            stroke_headers, stroke_Vabs, stroke_Pabs, stroke_Feats,
            label=1,  # 'stroke' label, not saved
            num_samples=NUM_SAMPLES,
            augment=AUGMENT_DATA
        )
    )
    
    # Save the processed stroke data
    with h5py.File(STROKE_OUTPUT_FILENAME, 'w') as f:
        save_variable_length_data(f, 'Vabs', s_Vabs_data)
        save_variable_length_data(f, 'Pabs', s_Pabs_data)
        save_variable_length_data(f, 'Feats', s_Feats_data)
        save_variable_length_data(f, 'headers', s_headers_data)
        save_variable_length_data(f, 'directions', s_directions_data)
        save_variable_length_data(f, 'orig_lengths', s_lengths_data)

   
#%% Save data to .ts files, univariate

import os
# os.chdir("D:/OneDrive - Queen's University/UNITS/weak")
# os.chdir("D:/OneDrive - Queen's University/MOMENT/moment-main")
os.chdir(r"D:\OneDrive - Queen's University\UNITS\For Nooshin\latest")

import h5py
import numpy as np
import csv

# from aeon import datasets
from sklearn.model_selection import train_test_split
from scipy.signal import resample

# Function to load variable length data from HDF5
def load_variable_length_data(h5file, dataset_name):
    grp = h5file[dataset_name]
    data = []
    for key in grp.keys():
        data.append(grp[key][:])
    return data

# Load selected data for control
with h5py.File('selected_control_raw_VandP_Pcorrected.h5', 'r') as f:
    control_Vabs_data = load_variable_length_data(f, 'Vabs')
    control_Pabs_data = load_variable_length_data(f, 'Pabs')
    control_Feats_data = load_variable_length_data(f, 'Feats')
    control_headers = load_variable_length_data(f, 'headers')
    # control_selected_indices = load_variable_length_data(f, 'selected_indices')
    control_directions = load_variable_length_data(f, 'directions')
    
    

# Load selected data for stroke
with h5py.File('selected_stroke_raw_VandP_Pcorrected.h5', 'r') as f:
    stroke_Vabs_data = load_variable_length_data(f, 'Vabs')
    stroke_Pabs_data = load_variable_length_data(f, 'Pabs')
    stroke_Feats_data = load_variable_length_data(f, 'Feats')
    stroke_headers = load_variable_length_data(f, 'headers')
    # stroke_selected_indices = load_variable_length_data(f, 'selected_indices')
    stroke_directions = load_variable_length_data(f, 'directions')


# Function to convert lists of ASCII values back to strings
def ascii_to_str(ascii_list):
    return ''.join([chr(c) for c in ascii_list])

# Convert loaded headers back to proper strings
def convert_headers(headers):
    converted_headers = []
    for header in headers:
        converted_header = []
        for field in header:
            if isinstance(field, np.ndarray):
                converted_header.append(ascii_to_str(field))
            else:
                converted_header.append(field)
        converted_headers.append(converted_header)
    return converted_headers

# Function to decode headers
def decode_headers(headers):
    decoded_headers = []
    for header in headers:
        decoded_header = [field.decode('utf-8') if isinstance(field, bytes) else field for field in header]
        decoded_headers.append(decoded_header)
    return decoded_headers

# def conver_to_ts (data, y, data_folder, ts_file_name):
#     datasets.write_to_tsfile(data, path=data_folder, y=y, problem_name=ts_file_name)


control_headers = decode_headers(convert_headers(control_headers))
stroke_headers = decode_headers(convert_headers(stroke_headers))




def resample_sections(data, num_sections=9, resample_to=64):
    section_length = data.shape[1] // num_sections
    resampled_data = []
    
    for i in range(num_sections):
        section = data[:, i * section_length: (i + 1) * section_length]
        resampled_section = resample(section, resample_to, axis=1)
        resampled_data.append(resampled_section)
    
    return np.concatenate(resampled_data, axis=1)

# Resampling control Pabs data
a = resample_sections(np.array(control_Pabs_data))


# Resampling control Vabs data
b = resample_sections(np.array(control_Vabs_data))


# Resampling stroke Pabs data
c = resample_sections(np.array(stroke_Pabs_data))


# Resampling stroke Vabs data
d = resample_sections(np.array(stroke_Vabs_data))





# Combine data
Vabs_data = np.concatenate([b, d])
Pabs_data = np.concatenate([a, c])
# Create labels: 0 for control, 1 for stroke
control_labels = np.zeros(len(control_Pabs_data), dtype=int)
stroke_labels = np.ones(len(stroke_Pabs_data), dtype=int)

# Concatenate labels
labels = np.concatenate([control_labels, stroke_labels])
headers = control_headers + stroke_headers
features = control_Feats_data + stroke_Feats_data
selected_indices = control_selected_indices + stroke_selected_indices
all_directions = control_directions + stroke_directions

# Split into train and test sets
Vabs_train_data, Vabs_test_data, Vabs_train_labels, Vabs_test_labels, Vabs_train_headers, Vabs_test_headers, Vabs_train_features, Vabs_test_features, Vabs_train_selected_indices, Vabs_test_selected_indices, Vabs_train_directions, Vabs_test_directions = train_test_split(
    Vabs_data, labels, headers, features, selected_indices, all_directions, test_size=0.3, random_state=42)

Pabs_train_data, Pabs_test_data, Pabs_train_labels, Pabs_test_labels, Pabs_train_headers, Pabs_test_headers, Pabs_train_features, Pabs_test_features, Pabs_train_selected_indices, Pabs_test_selected_indices, Pabs_train_directions, Pabs_test_directions = train_test_split(
    Pabs_data, labels, headers, features, selected_indices, all_directions, test_size=0.3, random_state=42)


# Function to save data to .ts file
def save_to_tsfile(data, labels, filename):
    # Determine the series length from the first element in the data
    if len(data) > 0:
        series_length = len(data[0])
    else:
        series_length = 0  # Handle empty data case

    with open(filename, 'w') as f:
        f.write("@problemName VGR\n")
        f.write("@timestamps false\n")
        f.write("@missing false\n")
        f.write("@univariate true\n")
        f.write("@equalLength true\n")
        f.write(f"@seriesLength {series_length}\n")  # Use dynamic series length
        # f.write("@classLabel true 0 1\n")
        f.write("@classLabel false\n")
        f.write("@data\n")
        
        for i in range(len(data)):
            series = data[i]
            label = labels[i]
            # f.write(f"{','.join(map(str, series))}:{label}\n")
            f.write(f"{','.join(map(str, series))}\n")


# # Function to save headers to a .csv file
# def save_headers(headers, filename):
#     with open(filename, 'w', newline='') as f:
#         writer = csv.writer(f)
#         # writer.writerow(['SubjectID', 'Hand', 'Task', 'DateTime'])
#         for header_group in headers:
#             for header in header_group:
#                 writer.writerow(header)

# Specify the directory where you have write permissions
output_dir = 'vgr_raw_ts'
os.makedirs(output_dir, exist_ok=True)

# Save Vabs and Pabs data to .ts files
save_to_tsfile(Vabs_train_data, Vabs_train_labels, os.path.join(output_dir, 'vgr_Vabs_TRAIN.ts'))
save_to_tsfile(Vabs_test_data, Vabs_test_labels, os.path.join(output_dir, 'vgr_Vabs_TEST.ts'))
save_to_tsfile(Pabs_train_data, Pabs_train_labels, os.path.join(output_dir, 'vgr_Pabs_train.ts'))
save_to_tsfile(Pabs_test_data, Pabs_test_labels, os.path.join(output_dir, 'vgr_Pabs_test.ts'))

# # Save headers to .csv files
# save_headers(Vabs_train_headers, os.path.join(output_dir, 'vgr_Vabs_train_headers.csv'))
# save_headers(Vabs_test_headers, os.path.join(output_dir, 'vgr_Vabs_test_headers.csv'))
# save_headers(Pabs_train_headers, os.path.join(output_dir, 'vgr_Pabs_train_headers.csv'))
# save_headers(Pabs_test_headers, os.path.join(output_dir, 'vgr_Pabs_test_headers.csv'))#%% Load data

np.save(os.path.join(output_dir,'vgr_Vabs_train_headers.npy'), Vabs_train_headers)
np.save(os.path.join(output_dir, 'vgr_Vabs_test_headers.npy'), Vabs_test_headers)
np.save(os.path.join(output_dir, 'vgr_Pabs_train_headers.npy'), Pabs_train_headers)
np.save(os.path.join(output_dir, 'vgr_Pabs_test_headers.npy'), Pabs_test_headers)

np.save(os.path.join(output_dir,'vgr_Vabs_train_features.npy'), Vabs_train_features)
np.save(os.path.join(output_dir, 'vgr_Vabs_test_features.npy'), Vabs_test_features)
np.save(os.path.join(output_dir, 'vgr_Pabs_train_features.npy'), Pabs_train_features)
np.save(os.path.join(output_dir, 'vgr_Pabs_test_features.npy'), Pabs_test_features)

np.save(os.path.join(output_dir,'vgr_Vabs_train_selected_indices.npy'), Vabs_train_selected_indices)
np.save(os.path.join(output_dir, 'vgr_Vabs_test_selected_indices.npy'), Vabs_test_selected_indices)
np.save(os.path.join(output_dir, 'vgr_Pabs_train_selected_indices.npy'), Pabs_train_selected_indices)
np.save(os.path.join(output_dir, 'vgr_Pabs_test_selected_indices.npy'), Pabs_test_selected_indices)

np.save(os.path.join(output_dir,'vgr_Vabs_train_directions.npy'), Vabs_train_directions)
np.save(os.path.join(output_dir, 'vgr_Vabs_test_directions.npy'), Vabs_test_directions)
np.save(os.path.join(output_dir, 'vgr_Pabs_train_directions.npy'), Pabs_train_directions)
np.save(os.path.join(output_dir, 'vgr_Pabs_test_directions.npy'), Pabs_test_directions)



#%% save data to .ts files, univariate, for Nooshin - 2

import os
os.chdir("D:/OneDrive - Queen's University/MOMENT/moment-main")
import h5py
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from scipy.signal import resample

##############################################################################
# 1) Parameters: "control", "stroke", or "both"
##############################################################################
GROUP_TO_PROCESS = "control"  # options: "control", "stroke", "both"

CONTROL_INPUT_FILENAME = 'selected_control_raw_VandP_Pcorrected_cropped_Mar_2025_adv.h5'
STROKE_INPUT_FILENAME = 'selected_stroke_raw_VandP_Pcorrected_cropped_Mar_2025_adv.h5'



OUTPUT_DIR = "vgr_raw_ts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_SIZE = 0.3
RANDOM_STATE = 42

##############################################################################
# 2) Utility functions
##############################################################################

def load_variable_length_data(h5file, dataset_name):
    """Load a variable-length dataset from HDF5 (each sub-dataset is appended)."""
    grp = h5file[dataset_name]
    data = []
    for key in grp.keys():
        data.append(grp[key][:])
    return data

def ascii_to_str(ascii_list):
    """Helper to convert ASCII code list to a Python string."""
    return ''.join([chr(c) for c in ascii_list])

def convert_headers(headers):
    """
    Convert loaded headers from arrays of ASCII codes to strings if needed.
    E.g., each 'field' might be a numpy array of ASCII codes.
    """
    converted_headers = []
    for header in headers:
        converted_header = []
        for field in header:
            if isinstance(field, np.ndarray):
                converted_header.append(ascii_to_str(field))
            else:
                converted_header.append(field)
        converted_headers.append(converted_header)
    return converted_headers

def decode_headers(headers):
    """Decode any byte fields into UTF-8 strings, if present."""
    decoded_headers = []
    for header in headers:
        decoded_header = [
            field.decode('utf-8') if isinstance(field, bytes) else field
            for field in header
        ]
        decoded_headers.append(decoded_header)
    return decoded_headers

def save_to_tsfile(data, labels, filename):
    """
    Save time-series data to a .ts file in a typical format used by 
    some time-series toolkits. 
    """
    if len(data) == 0:
        # handle empty data case
        with open(filename, 'w') as f:
            f.write("@problemName VGR\n")
            f.write("@timestamps false\n")
            f.write("@missing false\n")
            f.write("@univariate true\n")
            f.write("@equalLength true\n")
            f.write(f"@seriesLength 0\n")
            f.write("@classLabel false\n")
            f.write("@data\n")
        return

    # Determine the series length from the first element in the data
    series_length = len(data[0])

    with open(filename, 'w') as f:
        f.write("@problemName VGR\n")
        f.write("@timestamps false\n")
        f.write("@missing false\n")
        f.write("@univariate true\n")
        f.write("@equalLength true\n")
        f.write(f"@seriesLength {series_length}\n")
        f.write("@classLabel false\n")
        f.write("@data\n")

        for i in range(len(data)):
            series = data[i]
            f.write(f"{','.join(map(str, series))}\n")

def resample_sections(data, num_sections=9, resample_to=64):
    """
    Given an array of shape (n_samples, T), break each sample's time series
    into num_sections segments, resample each to length resample_to, 
    then concatenate them along axis=1.
    => final shape (n_samples, num_sections * resample_to).
    """
    section_length = data.shape[1] // num_sections
    resampled_data = []
    for i in range(num_sections):
        section = data[:, i * section_length:(i + 1) * section_length]
        resampled_section = resample(section, resample_to, axis=1)
        resampled_data.append(resampled_section)
    return np.concatenate(resampled_data, axis=1)

##############################################################################
# 3) Load data conditionally (control, stroke, or both)
##############################################################################

control_Vabs_data, control_Pabs_data = [], []
control_Feats_data, control_headers_data = [], []
control_directions_data = []
control_labels = np.array([], dtype=int)
control_lengths_data = []


if GROUP_TO_PROCESS in ["both", "control"]:
    with h5py.File(CONTROL_INPUT_FILENAME, "r") as f:
        control_Vabs_data = load_variable_length_data(f, 'Vabs')
        control_Pabs_data = load_variable_length_data(f, 'Pabs')
        control_Feats_data = load_variable_length_data(f, 'Feats')
        control_headers_data = load_variable_length_data(f, 'headers')
        control_directions_data = load_variable_length_data(f, 'directions')
        control_lengths_data = load_variable_length_data(f, 'orig_lengths')
    # Assign label 0 for these
    control_labels = np.zeros(len(control_Vabs_data), dtype=int)

stroke_Vabs_data, stroke_Pabs_data = [], []
stroke_Feats_data, stroke_headers_data = [], []
stroke_directions_data = []
stroke_labels = np.array([], dtype=int)
stroke_lengths_data = []

if GROUP_TO_PROCESS in ["both", "stroke"]:
    with h5py.File(STROKE_INPUT_FILENAME, "r") as f:
        stroke_Vabs_data = load_variable_length_data(f, 'Vabs')
        stroke_Pabs_data = load_variable_length_data(f, 'Pabs')
        stroke_Feats_data = load_variable_length_data(f, 'Feats')
        stroke_headers_data = load_variable_length_data(f, 'headers')
        stroke_directions_data = load_variable_length_data(f, 'directions')
        stroke_lengths_data = load_variable_length_data(f, 'orig_lengths')
    # Assign label 1 for these
    stroke_labels = np.ones(len(stroke_Vabs_data), dtype=int)

##############################################################################
# 4) Combine whichever data was loaded
##############################################################################

all_Vabs_data = control_Vabs_data + stroke_Vabs_data
all_Pabs_data = control_Pabs_data + stroke_Pabs_data
all_features = control_Feats_data + stroke_Feats_data

# Convert + decode headers
control_headers_data = decode_headers(convert_headers(control_headers_data))
stroke_headers_data  = decode_headers(convert_headers(stroke_headers_data))
all_headers = control_headers_data + stroke_headers_data

all_directions = control_directions_data + stroke_directions_data

all_labels = np.concatenate([control_labels, stroke_labels])

all_lengths   = control_lengths_data + stroke_lengths_data

##############################################################################
# 5) If there's no data at all, handle that case gracefully
##############################################################################
if len(all_Vabs_data) == 0:
    print("No data loaded for the specified group(s). Exiting.")
    # You could exit here or handle differently
    exit()

##############################################################################
# 6) Resample data into uniform shape
##############################################################################
# all_Vabs_data and all_Pabs_data are lists of arrays (each shape: (time_length,))
# We must ensure they have the same length so we can stack them.

vabs_lengths = [arr.shape[0] for arr in all_Vabs_data]
pabs_lengths = [arr.shape[0] for arr in all_Pabs_data]

if len(set(vabs_lengths)) == 1:
    # can stack
    vabs_array = np.vstack(all_Vabs_data)
else:
    raise ValueError("Not all Vabs time series have the same length. Cannot stack.")

if len(set(pabs_lengths)) == 1:
    # can stack
    pabs_array = np.vstack(all_Pabs_data)
else:
    raise ValueError("Not all Pabs time series have the same length. Cannot stack.")

# Resample into shape (N, num_sections * resample_to)
Vabs_data_resampled = resample_sections(vabs_array, num_sections=9, resample_to=64)
Pabs_data_resampled = resample_sections(pabs_array, num_sections=9, resample_to=64)

##############################################################################
# 7) Subject-level split (no data leakage)
##############################################################################
# Let’s assume subject ID is stored in header[0] for each trial
subject_ids = [h[0] for h in all_headers]  # adjust if needed
subject_ids = np.array(subject_ids, dtype=object)

unique_subjs = np.unique(subject_ids)

# If there's only one subject or no subjects, handle carefully:
if len(unique_subjs) == 0:
    print("No valid subject IDs found. Exiting.")
    exit()
elif len(unique_subjs) == 1:
    # Possibly we can't do a train_test_split, so everything goes to "train"
    train_subjs = unique_subjs
    test_subjs = []
else:
    # Normal subject-level train/test split
    train_subjs, test_subjs = train_test_split(
        unique_subjs, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

# Build boolean masks
train_mask = np.isin(subject_ids, train_subjs)
test_mask = np.isin(subject_ids, test_subjs)

# Slice everything
Vabs_train_data = Vabs_data_resampled[train_mask]
Vabs_test_data  = Vabs_data_resampled[test_mask]

Pabs_train_data = Pabs_data_resampled[train_mask]
Pabs_test_data  = Pabs_data_resampled[test_mask]

train_labels = all_labels[train_mask]
test_labels  = all_labels[test_mask]

train_headers = np.array(all_headers, dtype=object)[train_mask]
test_headers  = np.array(all_headers, dtype=object)[test_mask]

train_features = np.array(all_features, dtype=object)[train_mask]
test_features  = np.array(all_features, dtype=object)[test_mask]

train_directions = np.array(all_directions, dtype=object)[train_mask]
test_directions  = np.array(all_directions, dtype=object)[test_mask]

all_lengths = np.array(all_lengths, dtype=object)  # each item is shape (9,)

train_lengths = all_lengths[train_mask]
test_lengths  = all_lengths[test_mask]

##############################################################################
# 8) Save final .ts files
##############################################################################
save_to_tsfile(Vabs_train_data, train_labels, os.path.join(OUTPUT_DIR, "vgr_Vabs_TRAIN.ts"))
save_to_tsfile(Vabs_test_data,  test_labels,  os.path.join(OUTPUT_DIR, "vgr_Vabs_TEST.ts"))
save_to_tsfile(Pabs_train_data, train_labels, os.path.join(OUTPUT_DIR, "vgr_Pabs_train.ts"))
save_to_tsfile(Pabs_test_data,  test_labels,  os.path.join(OUTPUT_DIR, "vgr_Pabs_test.ts"))

##############################################################################
# 9) Save metadata (headers, features, directions, etc.) as .npy
##############################################################################
np.save(os.path.join(OUTPUT_DIR, "vgr_Vabs_train_headers.npy"), train_headers)
np.save(os.path.join(OUTPUT_DIR, "vgr_Vabs_test_headers.npy"),  test_headers)
np.save(os.path.join(OUTPUT_DIR, "vgr_Vabs_train_features.npy"), train_features)
np.save(os.path.join(OUTPUT_DIR, "vgr_Vabs_test_features.npy"),  test_features)
np.save(os.path.join(OUTPUT_DIR, "vgr_Vabs_train_directions.npy"), train_directions)
np.save(os.path.join(OUTPUT_DIR, "vgr_Vabs_test_directions.npy"),  test_directions)

# (Similarly for Pabs if you prefer separate naming conventions.)
np.save(os.path.join(OUTPUT_DIR, "vgr_Pabs_train_headers.npy"), train_headers)
np.save(os.path.join(OUTPUT_DIR, "vgr_Pabs_test_headers.npy"),  test_headers)
np.save(os.path.join(OUTPUT_DIR, "vgr_Pabs_train_features.npy"), train_features)
np.save(os.path.join(OUTPUT_DIR, "vgr_Pabs_test_features.npy"),  test_features)
np.save(os.path.join(OUTPUT_DIR, "vgr_Pabs_train_directions.npy"), train_directions)
np.save(os.path.join(OUTPUT_DIR, "vgr_Pabs_test_directions.npy"), test_directions)


np.save(os.path.join(OUTPUT_DIR, "vgr_train_orig_lengths.npy"), train_lengths)
np.save(os.path.join(OUTPUT_DIR, "vgr_test_orig_lengths.npy"),  test_lengths)

print(f"Done. Created train/test split at subject level for {GROUP_TO_PROCESS} data.")


#%%%%%%% Save data to .ts files, multivariate

# import os
# os.chdir("D:/OneDrive - Queen's University/UNITS/weak")

# import h5py
# import numpy as np
# import csv

# from aeon import datasets
# from sklearn.model_selection import train_test_split


# # Function to load variable length data from HDF5
# def load_variable_length_data(h5file, dataset_name):
#     grp = h5file[dataset_name]
#     data = []
#     for key in grp.keys():
#         data.append(grp[key][:])
#     return data

# # Load selected data for control
# with h5py.File('selected_control_raw_VandP_Pcorrected.h5', 'r') as f:
#     control_Vabs_data = load_variable_length_data(f, 'Vabs')
#     control_Pabs_data = load_variable_length_data(f, 'Pabs')
#     control_Feats_data = load_variable_length_data(f, 'Feats')
#     control_headers = load_variable_length_data(f, 'headers')

# # Load selected data for stroke
# with h5py.File('selected_stroke_raw_VandP_Pcorrected.h5', 'r') as f:
#     stroke_Vabs_data = load_variable_length_data(f, 'Vabs')
#     stroke_Pabs_data = load_variable_length_data(f, 'Pabs')
#     stroke_Feats_data = load_variable_length_data(f, 'Feats')
#     stroke_headers = load_variable_length_data(f, 'headers')

# # Function to convert lists of ASCII values back to strings
# def ascii_to_str(ascii_list):
#     return ''.join([chr(c) for c in ascii_list])

# # Convert loaded headers back to proper strings
# def convert_headers(headers):
#     converted_headers = []
#     for header in headers:
#         converted_header = []
#         for field in header:
#             if isinstance(field, np.ndarray):
#                 converted_header.append(ascii_to_str(field))
#             else:
#                 converted_header.append(field)
#         converted_headers.append(converted_header)
#     return converted_headers

# # Function to decode headers
# def decode_headers(headers):
#     decoded_headers = []
#     for header in headers:
#         decoded_header = [field.decode('utf-8') if isinstance(field, bytes) else field for field in header]
#         decoded_headers.append(decoded_header)
#     return decoded_headers

# def conver_to_ts (data, y, data_folder, ts_file_name):
#     datasets.write_to_tsfile(data, path=data_folder, y=y, problem_name=ts_file_name)

# control_headers = decode_headers(convert_headers(control_headers))
# stroke_headers = decode_headers(convert_headers(stroke_headers))


# # Combine data for both channels
# Vabs_data = np.concatenate([control_Vabs_data, stroke_Vabs_data])
# Pabs_data = np.concatenate([control_Pabs_data, stroke_Pabs_data])
# # Create labels: 0 for control, 1 for stroke
# control_labels = np.zeros(len(control_Pabs_data), dtype=int)
# stroke_labels = np.ones(len(stroke_Pabs_data), dtype=int)

# labels = np.concatenate([control_labels, stroke_labels])
# headers = control_headers + stroke_headers

# # Combine Vabs and Pabs into multichannel data
# combined_data = np.stack((Vabs_data, Pabs_data), axis=1)  # Shape: (n_samples, 2, series_length)

# # Split into train and test sets
# train_data, test_data, train_labels, test_labels, train_headers, test_headers = train_test_split(
#     combined_data, labels, headers, test_size=0.2, random_state=42)

# # Function to save multichannel data to a .ts file
# def save_to_tsfile(data, labels, filename):
#     # Determine the series length from the first element in the data
#     if len(data) > 0:
#         series_length = len(data[0])
#     else:
#         series_length = 0  # Handle empty data case
        
#     with open(filename, 'w') as f:
#         f.write("@problemName VGR\n")
#         f.write("@timestamps false\n")
#         f.write("@missing false\n")
#         f.write("@univariate false\n")  # Indicate multivariate data
#         f.write("@equalLength true\n")
#         f.write(f"@seriesLength {series_length}\n")  # Use dynamic series length
#         f.write("@classLabel true 0 1\n")
#         f.write("@data\n")
#         for i in range(len(data)):
#             series = data[i]
#             label = labels[i]
#             # Format each channel and join them with a comma
#             series_str = ','.join(['(' + ':'.join(map(str, channel)) + ')' for channel in series])
#             f.write(f"{series_str}:{label}\n")

# # # Function to save headers to a .csv file
# # def save_headers(headers, filename):
# #     with open(filename, 'w', newline='') as f:
# #         writer = csv.writer(f)
# #         # writer.writerow(['SubjectID', 'Hand', 'Task', 'DateTime'])
# #         for header_group in headers:
# #             for header in header_group:
# #                 writer.writerow(header)

# # Specify the directory where you have write permissions
# output_dir = 'vgr_raw_ts'
# os.makedirs(output_dir, exist_ok=True)

# # Save combined multichannel data to .ts files
# save_to_tsfile(train_data, train_labels, os.path.join(output_dir, 'vgr_TRAIN.ts'))
# save_to_tsfile(test_data, test_labels, os.path.join(output_dir, 'vgr_TEST.ts'))

# # Save headers to .csv files
# # save_headers(train_headers, os.path.join(output_dir, 'vgr_train_headers.csv'))
# # save_headers(test_headers, os.path.join(output_dir, 'vgr_test_headers.csv'))

# np.save(os.path.join(output_dir,'vgr_train_headers.npy'), train_headers)
# np.save(os.path.join(output_dir, 'vgr_test_headers.npy'), test_headers)






#%%%
#%% Updated Code to Save All Trials Using np.save:
import numpy as np
from scipy.signal import resample
import os
import h5py
import pickle

# os.chdir("D:/OneDrive - Queen's University/UNITS/weak")
os.chdir("D:/OneDrive - Queen's University/MOMENT/moment-main")

# Load control data from HDF5
with h5py.File('new_control_raw_VandP_Pcorrected_cropped_Mar_2025_adv.h5', 'r') as f:
    control_headers = [f['all_header_CT'][str(i)][:] for i in range(len(f['all_header_CT']))]
    control_Feats = [f['all_features_CT'][str(i)][:] for i in range(len(f['all_features_CT']))]
    control_Vabs = [f['all_Vabs_CT'][str(i)][:] for i in range(len(f['all_Vabs_CT']))]
    control_Pabs = [f['all_Pabs_CT'][str(i)][:] for i in range(len(f['all_Pabs_CT']))]

# Load stroke data from HDF5
with h5py.File('new_stroke_raw_VandP_Pcorrected_cropped_Mar_2025_adv.h5', 'r') as f:
    stroke_headers = [f['all_header_ST'][str(i)][:] for i in range(len(f['all_header_ST']))]
    stroke_Feats = [f['all_features_ST'][str(i)][:] for i in range(len(f['all_features_ST']))]
    stroke_Vabs = [f['all_Vabs_ST'][str(i)][:] for i in range(len(f['all_Vabs_ST']))]
    stroke_Pabs = [f['all_Pabs_ST'][str(i)][:] for i in range(len(f['all_Pabs_ST']))]

# Function to decode headers
def decode_headers(headers):
    decoded_headers = []
    for header in headers:
        decoded_header = [field.decode('utf-8') if isinstance(field, bytes) else field for field in header]
        decoded_headers.append(decoded_header)
    return decoded_headers

# Decode headers
control_headers = decode_headers(control_headers)
stroke_headers = decode_headers(stroke_headers)

# Function to extract all trials for a subject
def extract_all_trials(headers, Vabs, Pabs, Feats, label, num_samples: int = 64,
    resample_trials: bool = True):
    
    # Direction mapping if header[-3] is 1
    direction_map = {'4': '8', '3': '7', '2': '6', '9': '5', '8': '4', '7': '3', '6': '2', '5': '9'}
     
    unique_keys = {}
    for i, header in enumerate(headers):
        key = (header[0], header[2], header[-1].replace('.zip', ''))  # no need to decode as it's already handled
        direction = header[1]
        
        if header[-3] == '1' and direction in direction_map:
            direction = direction_map[direction]  # Map direction if needed 
            
        if key not in unique_keys:
            unique_keys[key] = {'Vabs': [], 'Pabs': [], 'Feats': [], 'headers': [], 'original_lengths': [], 'directions': []}
        unique_keys[key]['Vabs'].append(Vabs[i])
        unique_keys[key]['Pabs'].append(Pabs[i])
        unique_keys[key]['Feats'].append(Feats[i])
        unique_keys[key]['headers'].append(header)
        
        # Store the original length of each trial before resampling
        unique_keys[key]['original_lengths'].append(len(Vabs[i]))
        
        unique_keys[key]['directions'].append(direction)  # Store the direction
    
    selected_Vabs = []
    selected_Pabs = []
    selected_Feats = []
    selected_headers = []
    labels = []
    original_lengths = []
    all_directions = []  # Store all directions for each subject
    
    # NEW: We also create lists to store the representative trials per direction
    rep_Vabs_data = []   # shape = (#subject-sessions,) each item is a dict direction->(num_samples,)
    rep_Pabs_data = []   # same shape, for position
    
    for key, data in unique_keys.items():
        if resample_trials:
            # Select and downsample all trials
            ds_Vabs = [resample(v, num_samples) for v in data['Vabs']]
            ds_Pabs = [resample(p, num_samples) for p in data['Pabs']]
            
        else:
            ds_Vabs = [np.copy(v) for v in data['Vabs']]
            ds_Pabs = [np.copy(p) for p in data['Pabs']]
        
        selected_Vabs.append(ds_Vabs)
        selected_Pabs.append(ds_Pabs)
        
        selected_Feats.append(data['Feats'][0])  # Only save features from the first trial
        selected_headers.append(data['headers'][0])  # Only save headers from the first trial
        labels.append(label)
        
        # Store the list of original lengths for each trial in this set
        original_lengths.append(data['original_lengths'])
        
        all_directions.append(data['directions'])  # Store all directions
        
        # # (E) Build representative trial(s):
        # #     group ds_Vabs/ds_Pabs by direction => average them
        # direction_groups_v = {}
        # direction_groups_p = {}
        
        # for i_trial, d_str in enumerate(data['directions']):
        #     # e.g. d_str might be '4', '8', etc.
        #     if d_str not in direction_groups_v:
        #         direction_groups_v[d_str] = []
        #         direction_groups_p[d_str] = []
        #     direction_groups_v[d_str].append(ds_Vabs[i_trial])
        #     direction_groups_p[d_str].append(ds_Pabs[i_trial])
        
        # # Now average them:
        # rep_v_dict = {}
        # rep_p_dict = {}
        
        # for d_str, v_list in direction_groups_v.items():
        #     # v_list is e.g. [ array((64,)), array((64,)), ... ]
        #     # We want the elementwise mean:
        #     stacked_v = np.vstack(v_list)  # shape (n_trials_for_that_dir, 64)
        #     mean_v = np.mean(stacked_v, axis=0)  # shape (64,)
        #     rep_v_dict[d_str] = mean_v
            
        # for d_str, p_list in direction_groups_p.items():
        #     stacked_p = np.vstack(p_list)  # shape (n_trials_for_that_dir, 64)
        #     mean_p = np.mean(stacked_p, axis=0)
        #     rep_p_dict[d_str] = mean_p
        
        
        # # Store these dicts in our main list
        # rep_Vabs_data.append(rep_v_dict)
        # rep_Pabs_data.append(rep_p_dict)
        
        
        # (E) Build per-direction averages *only if* all trials share one length
        dirs = data['directions']
        lengths = [len(x) for x in ds_Vabs]
        if len(set(lengths)) == 1:
            # group by direction
            dir_groups_v = {}
            dir_groups_p = {}
            for i_trial, d_str in enumerate(dirs):
                dir_groups_v.setdefault(d_str, []).append(ds_Vabs[i_trial])
                dir_groups_p.setdefault(d_str, []).append(ds_Pabs[i_trial])
            # average each group
            rep_v = {
                d: np.mean(np.vstack(lst), axis=0)
                for d, lst in dir_groups_v.items()
            }
            rep_p = {
                d: np.mean(np.vstack(lst), axis=0)
                for d, lst in dir_groups_p.items()
            }
        else:
            # lengths differ → skip averaging
            rep_v, rep_p = {}, {}
        
        rep_Vabs_data.append(rep_v)
        rep_Pabs_data.append(rep_p)
        
        
    
    return selected_Vabs, selected_Pabs, selected_Feats, selected_headers, original_lengths, all_directions, rep_Vabs_data, rep_Pabs_data

# --------------------------------------------------------------------------
# 4) Extract & Pickle: Control
# --------------------------------------------------------------------------
control_Vabs_data, control_Pabs_data, control_Feats_data, control_headers, \
control_lengths, control_directions, control_repV, control_repP = \
    extract_all_trials(control_headers, control_Vabs, control_Pabs, 
                       control_Feats, label=0, num_samples=64, resample_trials=False)

# --------------------------------------------------------------------------
# 5) Extract & Pickle: Stroke
# --------------------------------------------------------------------------
stroke_Vabs_data, stroke_Pabs_data, stroke_Feats_data, stroke_headers, \
stroke_lengths, stroke_directions, stroke_repV, stroke_repP = \
    extract_all_trials(stroke_headers, stroke_Vabs, stroke_Pabs, 
                       stroke_Feats, label=1, num_samples=64, resample_trials=False)


# --------------------------------------------------------------------------
# 6) Save Everything with Pickle
# --------------------------------------------------------------------------
def save_data_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

# -- Control
save_data_pickle('control_Vabs_data_Mar_2025_nres.pkl',       control_Vabs_data)
save_data_pickle('control_Pabs_data_Mar_2025_nres.pkl',       control_Pabs_data)
save_data_pickle('control_Feats_data_Mar_2025_nres.pkl',      control_Feats_data)
save_data_pickle('control_headers_Mar_2025_nres.pkl',         control_headers)
save_data_pickle('control_lengths_Mar_2025_nres.pkl',         control_lengths)
save_data_pickle('control_directions_Mar_2025_nres.pkl',      control_directions)

# These 2 new pickles store the representative (average) trial by direction
save_data_pickle('control_repVabs_Mar_2025_nres.pkl',         control_repV)   # a list of dicts
save_data_pickle('control_repPabs_Mar_2025_nres.pkl',         control_repP)   # a list of dicts

# -- Stroke
save_data_pickle('stroke_Vabs_data_Mar_2025_nres.pkl',        stroke_Vabs_data)
save_data_pickle('stroke_Pabs_data_Mar_2025_nres.pkl',        stroke_Pabs_data)
save_data_pickle('stroke_Feats_data_Mar_2025_nres.pkl',       stroke_Feats_data)
save_data_pickle('stroke_headers_Mar_2025_nres.pkl',          stroke_headers)
save_data_pickle('stroke_lengths_Mar_2025_nres.pkl',          stroke_lengths)
save_data_pickle('stroke_directions_Mar_2025_nres.pkl',       stroke_directions)

save_data_pickle('stroke_repVabs_Mar_2025_nres.pkl',          stroke_repV)
save_data_pickle('stroke_repPabs_Mar_2025_nres.pkl',          stroke_repP)



# # Load data using pickle
# def load_data_pickle(file_name):
#     with open(file_name, 'rb') as f:
#         return pickle.load(f)
    
# # Example: Load control data
# control_Vabs_data = load_data_pickle('control_Vabs_data.pkl')
# control_Pabs_data = load_data_pickle('control_Pabs_data.pkl')
# control_Feats_data = load_data_pickle('control_Feats_data.pkl')
# control_headers = load_data_pickle('control_headers.pkl')

# # Similarly, load stroke data
# stroke_Vabs_data = load_data_pickle('stroke_Vabs_data.pkl')
# stroke_Pabs_data = load_data_pickle('stroke_Pabs_data.pkl')
# stroke_Feats_data = load_data_pickle('stroke_Feats_data.pkl')
# stroke_headers = load_data_pickle('stroke_headers.pkl')


#%% control test set

import os
os.chdir("D:/OneDrive - Queen's University/MOMENT/moment-main")
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------
# 1) Load your pickled data for the control group
# ----------------------------------------------------------------------
with open('control_Vabs_data_Mar_2025.pkl', 'rb') as f:
    control_Vabs_data = pickle.load(f)
with open('control_Pabs_data_Mar_2025.pkl', 'rb') as f:
    control_Pabs_data = pickle.load(f)
with open('control_Feats_data_Mar_2025.pkl', 'rb') as f:
    control_Feats_data = pickle.load(f)
with open('control_headers_Mar_2025.pkl', 'rb') as f:
    control_headers_data = pickle.load(f)
with open('control_lengths_Mar_2025.pkl', 'rb') as f:
    control_lengths_data = pickle.load(f)
with open('control_directions_Mar_2025.pkl', 'rb') as f:
    control_directions_data = pickle.load(f)

# Representative (average) trials, one dict per subject-session
with open('control_repVabs_Mar_2025.pkl', 'rb') as f:
    control_repVabs_data = pickle.load(f)
with open('control_repPabs_Mar_2025.pkl', 'rb') as f:
    control_repPabs_data = pickle.load(f)

# ----------------------------------------------------------------------
# 2) Identify the "subject-session" ID from the last element of each header
#    Then do a random 30% split
# ----------------------------------------------------------------------
session_ids = []
for hdr in control_headers_data:
    # last element might look like "ABC123.zip"
    # strip the ".zip" and treat that as session_id
    session_id = hdr[-1].replace('.zip', '')
    session_ids.append(session_id)

session_ids = np.array(session_ids, dtype=object)
unique_ids = np.unique(session_ids)

# 30% test split
train_ids, test_ids = train_test_split(unique_ids, test_size=0.3, random_state=42)

N = len(control_headers_data)
test_mask = np.isin(session_ids, test_ids)

# ----------------------------------------------------------------------
# 3) For each subject in the test set => pick 8 base trials, then append
#    the 8 rep trials => total 16. Then flatten to shape (16*64,)
# ----------------------------------------------------------------------
new_test_Vabs_list    = []  # Will store one flattened array per subject-session
new_test_Pabs_list    = []
new_test_Feats_list   = []
new_test_Headers_list = []
new_test_Lengths_list = []
new_test_Dir_list     = []

for i in range(N):
    if not test_mask[i]:
        # This subject-session is in the train portion, skip
        continue
    
    # Collect data for subject-session i
    all_v = control_Vabs_data[i]         # list of arrays, each shape (64,)
    all_p = control_Pabs_data[i]
    rep_v = control_repVabs_data[i]      # dict {dir_str -> array(64,)}
    rep_p = control_repPabs_data[i]
    
    feats     = control_Feats_data[i]
    header    = control_headers_data[i]
    orig_lens = control_lengths_data[i]
    all_dirs  = control_directions_data[i]  # list of direction strings
    
    all_dirs = np.array(all_dirs, dtype=object)
    unique_dirs = np.unique(all_dirs)
    num_dirs = len(unique_dirs)
    
    # Decide if 8- or 4-direction subject
    if num_dirs == 8:
        # Base 8 => first trial from each direction in the order encountered
        base_indices = []
        used_dirs = []
        for d_str in all_dirs:
            if d_str not in used_dirs:
                # index of the first occurrence
                idx = np.where(all_dirs == d_str)[0][0]
                base_indices.append(idx)
                used_dirs.append(d_str)
            if len(used_dirs) == 8:
                break
                
    elif num_dirs == 4:
        # Base 8 => first 2 trials from each direction
        base_indices = []
        dir_counts = {}
        for idx, d_str in enumerate(all_dirs):
            if d_str not in dir_counts:
                dir_counts[d_str] = 0
            if dir_counts[d_str] < 2:
                base_indices.append(idx)
                dir_counts[d_str] += 1
            if len(base_indices) == 8:
                break
        used_dirs = list(all_dirs[base_indices])
        
    else:
        # If not 4 or 8 directions, skip
        continue
    
    # Build the "base" velocity/position arrays
    base_v_list = [all_v[idx] for idx in base_indices]  # each shape(64,)
    base_p_list = [all_p[idx] for idx in base_indices]
    
    # Build the "rep" velocity/position arrays
    # The direction order is used_dirs. So we gather rep_v[d_str] in that order.
    rep_v_list = []
    rep_p_list = []
    for d_str in used_dirs:
        rep_v_list.append(rep_v[d_str])  # shape(64,)
        rep_p_list.append(rep_p[d_str])  # shape(64,)
    
    # Concatenate base & rep => shape(16,64) each
    v_concat = np.vstack(base_v_list + rep_v_list)
    p_concat = np.vstack(base_p_list + rep_p_list)
    
    # Flatten => shape(16*64,) = (1024,)
    v_flat = v_concat.flatten()
    p_flat = p_concat.flatten()
    
    # Build direction array of length 16
    dir_16 = used_dirs + used_dirs  # or np.concatenate if you want a np.array
    
    # Build length array
    # base 8 => from orig_lens for each index, rep 8 => all 64 or "rep"
    base_lengths = [orig_lens[idx] for idx in base_indices]
    rep_lengths  = [64]*len(used_dirs)  # since each rep is 64 after downsampling
    length_16 = base_lengths + rep_lengths
    
    # Store in our new test set
    new_test_Vabs_list.append(v_flat)       # shape (1024,)
    new_test_Pabs_list.append(p_flat)
    new_test_Feats_list.append(feats)       # single item from that subject
    new_test_Headers_list.append(header)    # single item
    new_test_Lengths_list.append(length_16) # list of length 16
    new_test_Dir_list.append(dir_16)        # list of length 16

# ----------------------------------------------------------------------
# 4) Convert new_test_Vabs_list, new_test_Pabs_list to arrays => shape (M, 1024)
# ----------------------------------------------------------------------
new_test_Vabs_arr = np.array(new_test_Vabs_list)  # shape (M, 1024)
new_test_Pabs_arr = np.array(new_test_Pabs_list)  # shape (M, 1024)

# The other items (directions, feats, headers, lengths) we keep as lists
# but we store them all in one dictionary:
new_test_set_control = {
    "Vabs": new_test_Vabs_arr,      # (M,1024)
    "Pabs": new_test_Pabs_arr,      # (M,1024)
    "Feats": new_test_Feats_list,   # length=M
    "Headers": new_test_Headers_list,
    "Lengths": new_test_Lengths_list, 
    "Directions": new_test_Dir_list
}

# ----------------------------------------------------------------------
# 5) Save as a pickle
# ----------------------------------------------------------------------
with open("new_test_set_control_Mar_2025.pkl", "wb") as f:
    pickle.dump(new_test_set_control, f)

print("Saved new_test_set_control_Mar_2025.pkl with shape:", new_test_set_control["Vabs"].shape)

#%% Load

import os
import pickle

# 1) Change working directory
os.chdir("D:/OneDrive - Queen's University/MOMENT/moment-main/data")

# 2) Load the pickle file
pickle_filename = "new_test_set_control.pkl"
with open(pickle_filename, "rb") as f:
    new_test_set_control = pickle.load(f)

# 3) Extract data
new_test_Vabs_arr       = new_test_set_control["Vabs"]       # shape (M, 1024)
new_test_Pabs_arr       = new_test_set_control["Pabs"]       # shape (M, 1024)
new_test_Feats_list      = new_test_set_control["Feats"]      # length = M
new_test_Headers_list    = new_test_set_control["Headers"]    # length = M
new_test_Lengths_list    = new_test_set_control["Lengths"]    # length = M (each item is length=16)
new_test_Dir_list = new_test_set_control["Directions"] # length = M (each item is length=16)

# 4) (Optional) Print shapes or a small sample
# print("Vabs shape:", new_test_Vabs_arr.shape)
# print("Pabs shape:", Pabs.shape)
# print("Number of subjects:", len(Headers))
# print("First subject's directions:", Directions[0])


#%% plot
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

def plot_16_trial_signal_with_resample(
    dataset_pickle, 
    subject_idx=0,
    do_show=True
):
    """
    Loads the specified pickled dataset (with shape (M, 16*64) for 'Vabs'),
    picks the subject_idx-th row, and plots the 16 trials as one continuous line.
    
    - The first 8 trials are resampled from 64 points -> their original length 
      (found in data["Lengths"][subject_idx][t]).
    - The last 8 "representative" trials are resampled from 64 -> the 
      average of the base trials that share the same direction.

    We draw vertical lines marking each trial boundary and label the 
    direction at the midpoint of that trial segment.
    """

    # 1) Load your dataset
    with open(dataset_pickle, "rb") as f:
        data = pickle.load(f)
    
    # data["Vabs"] => shape (M, 1024)
    # data["Directions"] => list of length M; each item => list of 16 directions
    # data["Lengths"] => list of length M; each item => list of 16 lengths
    # first 8 => original lengths, last 8 => 64 for reps
    # etc.
    
    all_Vabs = data["Vabs"]              # (M, 1024)
    all_dirs = data["Directions"]        # list of length M, each => 16 directions
    all_lens = data["Lengths"]           # same shape
    # Optional: headers = data["Headers"], feats = data["Feats"], if needed

    M = all_Vabs.shape[0]
    if subject_idx < 0 or subject_idx >= M:
        raise IndexError(f"Invalid subject_idx={subject_idx}. Must be in [0, {M-1}].")

    v_row   = all_Vabs[subject_idx]   # shape (1024,)
    dirs_row= all_dirs[subject_idx]   # list of 16 directions
    lens_row= all_lens[subject_idx]   # list of 16 lengths (or 64 for reps)

    # 2) Gather base original lengths by direction
    #    The first 8 trials are the "base" trials
    #    We'll store direction -> list of base original lengths
    dir_to_base_lengths = {}
    for t in range(8):
        d_str = dirs_row[t]
        orig_len = lens_row[t]
        if d_str not in dir_to_base_lengths:
            dir_to_base_lengths[d_str] = []
        dir_to_base_lengths[d_str].append(orig_len)

    # 3) Build a new array for the plotted signal
    #    We'll keep them in a list, then concatenate
    resampled_segments = []
    segment_lengths = []  # store each new length to help with boundary plotting

    for t in range(16):
        # Extract the 64-sample chunk
        start_idx = t * 64
        end_idx   = (t+1) * 64
        trial_64  = v_row[start_idx:end_idx]  # shape(64,)

        d_str = dirs_row[t]

        # Determine the new (original) length
        if t < 8:
            # base trial => lens_row[t]
            target_len = lens_row[t]
        else:
            # rep trial => average base length for that direction
            if d_str in dir_to_base_lengths and len(dir_to_base_lengths[d_str]) > 0:
                base_lens = dir_to_base_lengths[d_str]
                target_len = int(round(np.mean(base_lens)))
            else:
                # fallback if no base trial has that direction
                target_len = 64

        # Resample
        if target_len == 64:
            # no change
            upsampled = trial_64
        else:
            upsampled = resample(trial_64, target_len)
        
        resampled_segments.append(upsampled)
        segment_lengths.append(target_len)

    # 4) Concatenate all segments into one array for plotting
    final_signal = np.concatenate(resampled_segments)
    
    # We'll compute boundaries for each trial
    boundaries = np.cumsum([0] + segment_lengths)  # shape (17,)

    # 5) Plot
    plt.figure(figsize=(12, 5))
    plt.plot(final_signal, color="blue", label="Velocity")

    # Draw vertical lines at each boundary (excluding the very end if you like)
    for i in range(1, 16):
        x_pos = boundaries[i]
        plt.axvline(x=x_pos, color="gray", linestyle="--")
    
    # Label each trial's direction near its midpoint
    for i in range(16):
        d_str = dirs_row[i]
        left = boundaries[i]
        right= boundaries[i+1]
        midpoint = (left + right)/2.0
        # We'll place text 15% above the max amplitude
        max_val = np.max(final_signal)
        plt.text(midpoint, max_val * 0.85, str(d_str), 
                 ha="center", color="red", fontsize=12)

    # plt.xlabel("Sample index (resampled, variable total length)")
    plt.ylabel("Speed amplitude")
    plt.title(f"Subject index {subject_idx}\n(First 8 trials -> original lengths, Next 8 -> avg. base lengths)")
    plt.legend()
    plt.tight_layout()
    
    if do_show:
        plt.show()

# -----------------------------------------------------------------------------
# Example usage:
vv = np.random.randint(0, 203)
plot_16_trial_signal_with_resample("new_test_set_control.pkl", subject_idx=vv)
# -----------------------------------------------------------------------------

#%% control test set 2

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

os.chdir("D:/OneDrive - Queen's University/MOMENT/moment-main/data")

# ----------------------------------------------------------------------
# 1) Load your pickled data for the control group
# ----------------------------------------------------------------------
with open('control_Vabs_data.pkl', 'rb') as f:
    control_Vabs_data = pickle.load(f)
with open('control_Pabs_data.pkl', 'rb') as f:
    control_Pabs_data = pickle.load(f)
with open('control_Feats_data.pkl', 'rb') as f:
    control_Feats_data = pickle.load(f)
with open('control_headers.pkl', 'rb') as f:
    control_headers_data = pickle.load(f)
with open('control_lengths.pkl', 'rb') as f:
    control_lengths_data = pickle.load(f)
with open('control_directions.pkl', 'rb') as f:
    control_directions_data = pickle.load(f)

# Representative (average) data is NOT needed here, 
# but we'll load them to keep the code consistent:
with open('control_repVabs.pkl', 'rb') as f:
    control_repVabs_data = pickle.load(f)
with open('control_repPabs.pkl', 'rb') as f:
    control_repPabs_data = pickle.load(f)

# ----------------------------------------------------------------------
# 2) Identify the same test set (30%) as before
#    We must replicate EXACTLY how we chose session_ids in the prior code
# ----------------------------------------------------------------------
session_ids = []
for hdr in control_headers_data:
    session_id = hdr[-1].replace('.zip', '')
    session_ids.append(session_id)

session_ids = np.array(session_ids, dtype=object)
unique_ids = np.unique(session_ids)

# We recreate the same train/test split from the previous code.
# We use the same random_state=42 for consistency.
train_ids, test_ids = train_test_split(unique_ids, test_size=0.3, random_state=42)

N = len(control_headers_data)
test_mask = np.isin(session_ids, test_ids)

# ----------------------------------------------------------------------
# 3) Build the "new_test_set_control_2" with 8 base + 8 random
# ----------------------------------------------------------------------

new_test_Vabs_list_2    = []
new_test_Pabs_list_2    = []
new_test_Feats_list_2   = []
new_test_Headers_list_2 = []
new_test_Lengths_list_2 = []
new_test_Dir_list_2     = []

# We can fix the random seed so the random choice is reproducible:
rng = np.random.default_rng(42)

for i in range(N):
    if not test_mask[i]:
        # Only process the same test subjects
        continue
    
    # Gather data for subject-session i
    all_v = control_Vabs_data[i]   # list of arrays, each shape(64,)
    all_p = control_Pabs_data[i]
    
    feats     = control_Feats_data[i]
    header    = control_headers_data[i]
    orig_lens = control_lengths_data[i]  # list of original lengths
    all_dirs  = control_directions_data[i]  # list of directions for each trial

    all_dirs = np.array(all_dirs, dtype=object)
    unique_dirs = np.unique(all_dirs)
    num_dirs = len(unique_dirs)

    # Decide if subject has 8 or 4 directions => pick base 8 accordingly
    if num_dirs == 8:
        # Base 8 => first trial from each direction
        base_indices = []
        used_dirs = []
        for d_str in all_dirs:
            if d_str not in used_dirs:
                idx = np.where(all_dirs == d_str)[0][0]
                base_indices.append(idx)
                used_dirs.append(d_str)
            if len(used_dirs) == 8:
                break
    elif num_dirs == 4:
        # Base 8 => first 2 trials from each direction
        base_indices = []
        dir_counts = {}
        for idx, d_str in enumerate(all_dirs):
            if d_str not in dir_counts:
                dir_counts[d_str] = 0
            if dir_counts[d_str] < 2:
                base_indices.append(idx)
                dir_counts[d_str] += 1
            if len(base_indices) == 8:
                break
    else:
        # skip if not 4 or 8 directions
        continue
    
    # Now we gather these base 8 velocity/position trials
    base_v_list = [all_v[idx] for idx in base_indices]  # each shape(64,)
    base_p_list = [all_p[idx] for idx in base_indices]
    
    # Build directions, lengths for these 8 base trials
    base_directions = list(all_dirs[base_indices])
    base_lengths    = [orig_lens[idx] for idx in base_indices]

    # ------------------------------------------------------------------
    # 4) Choose 8 random trials from the entire set, ignoring the base 8
    #    Could skip duplicates or allow them. We'll skip duplicates.
    # ------------------------------------------------------------------
    n_all_trials = len(all_v)
    remaining_indices = [x for x in range(n_all_trials) if x not in base_indices]
    
    # If there aren't at least 8 leftover trials, skip or handle differently
    if len(remaining_indices) < 8:
        continue
    
    random_indices = rng.choice(remaining_indices, size=8, replace=False)
    
    # Gather these random velocity/position trials
    rand_v_list = [all_v[idx] for idx in random_indices]
    rand_p_list = [all_p[idx] for idx in random_indices]
    
    # Corresponding directions and lengths
    rand_directions = list(all_dirs[random_indices])
    rand_lengths    = [orig_lens[idx] for idx in random_indices]

    # ------------------------------------------------------------------
    # 5) Combine base + random => shape(16,64)
    # ------------------------------------------------------------------
    v_concat = np.vstack(base_v_list + rand_v_list)  # shape (16,64)
    p_concat = np.vstack(base_p_list + rand_p_list)
    
    # Flatten => shape(1024,)
    v_flat = v_concat.flatten()
    p_flat = p_concat.flatten()
    
    # Directions => first 8 from base, next 8 from random
    dir_16 = base_directions + rand_directions
    
    # Lengths => first 8 from base_lengths, next 8 from rand_lengths
    length_16 = base_lengths + rand_lengths

    # ------------------------------------------------------------------
    # 6) Append to our new lists
    # ------------------------------------------------------------------
    new_test_Vabs_list_2.append(v_flat)       # (1024,)
    new_test_Pabs_list_2.append(p_flat)       # (1024,)
    new_test_Feats_list_2.append(feats)       # single item
    new_test_Headers_list_2.append(header)    # single item
    new_test_Lengths_list_2.append(length_16) # (16,)
    new_test_Dir_list_2.append(dir_16)        # (16,)

# ----------------------------------------------------------------------
# 7) Convert the new lists to arrays => shape (M2, 1024)
# ----------------------------------------------------------------------
new_test_Vabs_arr_2 = np.array(new_test_Vabs_list_2)
new_test_Pabs_arr_2 = np.array(new_test_Pabs_list_2)

# We keep feats, headers, lengths, directions as lists
new_test_set_control_2 = {
    "Vabs": new_test_Vabs_arr_2,      # shape (M2, 1024)
    "Pabs": new_test_Pabs_arr_2,      # shape (M2, 1024)
    "Feats": new_test_Feats_list_2,   
    "Headers": new_test_Headers_list_2,
    "Lengths": new_test_Lengths_list_2, 
    "Directions": new_test_Dir_list_2
}

# ----------------------------------------------------------------------
# 8) Save as a pickle with "_2"
# ----------------------------------------------------------------------
with open("new_test_set_control_2.pkl", "wb") as f:
    pickle.dump(new_test_set_control_2, f)

print("Saved new_test_set_control_2.pkl with shape:", new_test_set_control_2["Vabs"].shape)


#%% Load

import os
import pickle

# 1) Change working directory
os.chdir("D:/OneDrive - Queen's University/MOMENT/moment-main/data")

# 2) Load the pickle file
pickle_filename = "new_test_set_control_2.pkl"
with open(pickle_filename, "rb") as f:
    new_test_set_control = pickle.load(f)

# 3) Extract data
new_test_Vabs_arr       = new_test_set_control["Vabs"]       # shape (M, 1024)
new_test_Pabs_arr       = new_test_set_control["Pabs"]       # shape (M, 1024)
new_test_Feats_list      = new_test_set_control["Feats"]      # length = M
new_test_Headers_list    = new_test_set_control["Headers"]    # length = M
new_test_Lengths_list    = new_test_set_control["Lengths"]    # length = M (each item is length=16)
new_test_Dir_list = new_test_set_control["Directions"] # length = M (each item is length=16)

# 4) (Optional) Print shapes or a small sample
# print("Vabs shape:", new_test_Vabs_arr.shape)
# print("Pabs shape:", Pabs.shape)
# print("Number of subjects:", len(Headers))
# print("First subject's directions:", Directions[0])


#%% plot
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

def plot_16_trial_signal_with_resample(
    dataset_pickle, 
    subject_idx=0,
    do_show=True
):
    """
    Loads the specified pickled dataset (with shape (M, 16*64) for 'Vabs'),
    picks the subject_idx-th row, and plots the 16 trials as one continuous line.
    
    - The first 8 trials are resampled from 64 points -> their original length 
      (found in data["Lengths"][subject_idx][t]).
    - The last 8 "representative" trials are resampled from 64 -> the 
      average of the base trials that share the same direction.

    We draw vertical lines marking each trial boundary and label the 
    direction at the midpoint of that trial segment.
    """

    # 1) Load your dataset
    with open(dataset_pickle, "rb") as f:
        data = pickle.load(f)
    
    # data["Vabs"] => shape (M, 1024)
    # data["Directions"] => list of length M; each item => list of 16 directions
    # data["Lengths"] => list of length M; each item => list of 16 lengths
    # first 8 => original lengths, last 8 => 64 for reps
    # etc.
    
    all_Vabs = data["Vabs"]              # (M, 1024)
    all_dirs = data["Directions"]        # list of length M, each => 16 directions
    all_lens = data["Lengths"]           # same shape
    # Optional: headers = data["Headers"], feats = data["Feats"], if needed

    M = all_Vabs.shape[0]
    if subject_idx < 0 or subject_idx >= M:
        raise IndexError(f"Invalid subject_idx={subject_idx}. Must be in [0, {M-1}].")

    v_row   = all_Vabs[subject_idx]   # shape (1024,)
    dirs_row= all_dirs[subject_idx]   # list of 16 directions
    lens_row= all_lens[subject_idx]   # list of 16 lengths (or 64 for reps)

    # 2) Gather base original lengths by direction
    #    The first 8 trials are the "base" trials
    #    We'll store direction -> list of base original lengths
    dir_to_base_lengths = {}
    for t in range(8):
        d_str = dirs_row[t]
        orig_len = lens_row[t]
        if d_str not in dir_to_base_lengths:
            dir_to_base_lengths[d_str] = []
        dir_to_base_lengths[d_str].append(orig_len)

    # 3) Build a new array for the plotted signal
    #    We'll keep them in a list, then concatenate
    resampled_segments = []
    segment_lengths = []  # store each new length to help with boundary plotting

    for t in range(16):
        # Extract the 64-sample chunk
        start_idx = t * 64
        end_idx   = (t+1) * 64
        trial_64  = v_row[start_idx:end_idx]  # shape(64,)

        d_str = dirs_row[t]

        # Determine the new (original) length
        if t < 8:
            # base trial => lens_row[t]
            target_len = lens_row[t]
        else:
            # rep trial => average base length for that direction
            if d_str in dir_to_base_lengths and len(dir_to_base_lengths[d_str]) > 0:
                base_lens = dir_to_base_lengths[d_str]
                target_len = int(round(np.mean(base_lens)))
            else:
                # fallback if no base trial has that direction
                target_len = 64

        # Resample
        if target_len == 64:
            # no change
            upsampled = trial_64
        else:
            upsampled = resample(trial_64, target_len)
        
        resampled_segments.append(upsampled)
        segment_lengths.append(target_len)

    # 4) Concatenate all segments into one array for plotting
    final_signal = np.concatenate(resampled_segments)
    
    # We'll compute boundaries for each trial
    boundaries = np.cumsum([0] + segment_lengths)  # shape (17,)

    # 5) Plot
    plt.figure(figsize=(12, 5))
    plt.plot(final_signal, color="blue", label="Velocity")

    # Draw vertical lines at each boundary (excluding the very end if you like)
    for i in range(1, 16):
        x_pos = boundaries[i]
        plt.axvline(x=x_pos, color="gray", linestyle="--")
    
    # Label each trial's direction near its midpoint
    for i in range(16):
        d_str = dirs_row[i]
        left = boundaries[i]
        right= boundaries[i+1]
        midpoint = (left + right)/2.0
        # We'll place text 15% above the max amplitude
        max_val = np.max(final_signal)
        plt.text(midpoint, max_val * 0.85, str(d_str), 
                 ha="center", color="white", fontsize=12)

    # plt.xlabel("Sample index (resampled, variable total length)")
    plt.ylabel("Speed amplitude")
    plt.title(f"Subject index {subject_idx}\n(First 8 trials -> original lengths, Next 8 -> avg. base lengths)")
    plt.legend()
    plt.tight_layout()
    
    if do_show:
        plt.show()

# -----------------------------------------------------------------------------
# Example usage:
plot_16_trial_signal_with_resample("new_test_set_control_2.pkl", subject_idx=165)
# -------


#%% control test set 3

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

os.chdir("D:/OneDrive - Queen's University/MOMENT/moment-main/data")

# ----------------------------------------------------------------------
# 1) Load your pickled data for the control group
# ----------------------------------------------------------------------
with open('control_Vabs_data.pkl', 'rb') as f:
    control_Vabs_data = pickle.load(f)
with open('control_Pabs_data.pkl', 'rb') as f:
    control_Pabs_data = pickle.load(f)
with open('control_Feats_data.pkl', 'rb') as f:
    control_Feats_data = pickle.load(f)
with open('control_headers.pkl', 'rb') as f:
    control_headers_data = pickle.load(f)
with open('control_lengths.pkl', 'rb') as f:
    control_lengths_data = pickle.load(f)
with open('control_directions.pkl', 'rb') as f:
    control_directions_data = pickle.load(f)

# We don't need representative trials here, but we can load them 
# to keep the environment consistent:
with open('control_repVabs.pkl', 'rb') as f:
    control_repVabs_data = pickle.load(f)
with open('control_repPabs.pkl', 'rb') as f:
    control_repPabs_data = pickle.load(f)

# ----------------------------------------------------------------------
# 2) Identify the same test set (30%) as before
#    We replicate EXACTLY the same session_ids logic as in prior code.
# ----------------------------------------------------------------------
session_ids = []
for hdr in control_headers_data:
    session_id = hdr[-1].replace('.zip', '')
    session_ids.append(session_id)

session_ids = np.array(session_ids, dtype=object)
unique_ids = np.unique(session_ids)

# Use the same random_state=42 so it's consistent with the previous test set
train_ids, test_ids = train_test_split(unique_ids, test_size=0.3, random_state=42)

N = len(control_headers_data)
test_mask = np.isin(session_ids, test_ids)

# ----------------------------------------------------------------------
# 3) Build "new_test_set_control_3": 8 base + 8 repeated random
# ----------------------------------------------------------------------
new_test_Vabs_list_3    = []
new_test_Pabs_list_3    = []
new_test_Feats_list_3   = []
new_test_Headers_list_3 = []
new_test_Lengths_list_3 = []
new_test_Dir_list_3     = []

rng = np.random.default_rng(42)  # For reproducible random selection

for i in range(N):
    if not test_mask[i]:
        # Only process the same test subjects
        continue
    
    # Gather data for subject-session i
    all_v = control_Vabs_data[i]   # list of arrays, each shape(64,)
    all_p = control_Pabs_data[i]
    
    feats     = control_Feats_data[i]
    header    = control_headers_data[i]
    orig_lens = control_lengths_data[i]  # list of original lengths
    all_dirs  = control_directions_data[i]  # list of directions for each trial

    all_dirs = np.array(all_dirs, dtype=object)
    unique_dirs = np.unique(all_dirs)
    num_dirs = len(unique_dirs)

    # Decide if subject has 8 or 4 directions => pick base 8 accordingly
    if num_dirs == 8:
        # Base 8 => first trial from each direction
        base_indices = []
        used_dirs = []
        for d_str in all_dirs:
            if d_str not in used_dirs:
                idx = np.where(all_dirs == d_str)[0][0]
                base_indices.append(idx)
                used_dirs.append(d_str)
            if len(used_dirs) == 8:
                break
    elif num_dirs == 4:
        # Base 8 => first 2 trials from each direction
        base_indices = []
        dir_counts = {}
        for idx, d_str in enumerate(all_dirs):
            if d_str not in dir_counts:
                dir_counts[d_str] = 0
            if dir_counts[d_str] < 2:
                base_indices.append(idx)
                dir_counts[d_str] += 1
            if len(base_indices) == 8:
                break
    else:
        # skip if not 4 or 8 directions
        continue
    
    # Now we gather these base 8 velocity/position trials
    base_v_list = [all_v[idx] for idx in base_indices]  # each shape(64,)
    base_p_list = [all_p[idx] for idx in base_indices]
    
    # Build directions, lengths for these 8 base trials
    base_directions = list(all_dirs[base_indices])
    base_lengths    = [orig_lens[idx] for idx in base_indices]

    # ------------------------------------------------------------------
    # 4) Choose 1 random trial from the leftover trials, then repeat it 8 times
    # ------------------------------------------------------------------
    n_all_trials = len(all_v)
    remaining_indices = [x for x in range(n_all_trials) if x not in base_indices]
    
    if len(remaining_indices) == 0:
        # If there are no leftover trials, we can't do the random pick
        # Skip or handle differently
        continue
    
    # Pick exactly 1 random leftover trial
    rand_idx = rng.choice(remaining_indices, size=1)[0]
    
    # We'll replicate that single trial 8 times
    repeated_v_list = [all_v[rand_idx]] * 8
    repeated_p_list = [all_p[rand_idx]] * 8

    repeated_directions = [all_dirs[rand_idx]] * 8
    repeated_lengths    = [orig_lens[rand_idx]] * 8  # keep the original length

    # ------------------------------------------------------------------
    # 5) Combine base + repeated => shape(16,64)
    # ------------------------------------------------------------------
    v_concat = np.vstack(base_v_list + repeated_v_list)  # shape (16,64)
    p_concat = np.vstack(base_p_list + repeated_p_list)
    
    # Flatten => shape(1024,)
    v_flat = v_concat.flatten()
    p_flat = p_concat.flatten()
    
    # Directions => first 8 from base, next 8 repeated random
    dir_16 = base_directions + repeated_directions
    
    # Lengths => first 8 from base_lengths, next 8 repeated
    length_16 = base_lengths + repeated_lengths

    # ------------------------------------------------------------------
    # 6) Append to our new lists
    # ------------------------------------------------------------------
    new_test_Vabs_list_3.append(v_flat)       # (1024,)
    new_test_Pabs_list_3.append(p_flat)       # (1024,)
    new_test_Feats_list_3.append(feats)       # single item
    new_test_Headers_list_3.append(header)    # single item
    new_test_Lengths_list_3.append(length_16) # (16,)
    new_test_Dir_list_3.append(dir_16)        # (16,)

# ----------------------------------------------------------------------
# 7) Convert lists to arrays => shape (M3, 1024)
# ----------------------------------------------------------------------
new_test_Vabs_arr_3 = np.array(new_test_Vabs_list_3)
new_test_Pabs_arr_3 = np.array(new_test_Pabs_list_3)

new_test_set_control_3 = {
    "Vabs": new_test_Vabs_arr_3,      # shape (M3, 1024)
    "Pabs": new_test_Pabs_arr_3,      # shape (M3, 1024)
    "Feats": new_test_Feats_list_3,   
    "Headers": new_test_Headers_list_3,
    "Lengths": new_test_Lengths_list_3, 
    "Directions": new_test_Dir_list_3
}

# ----------------------------------------------------------------------
# 8) Save as a pickle with "_3"
# ----------------------------------------------------------------------
with open("new_test_set_control_3.pkl", "wb") as f:
    pickle.dump(new_test_set_control_3, f)

print("Saved new_test_set_control_3.pkl with shape:", new_test_set_control_3["Vabs"].shape)


#%% Load

import os
import pickle

# 1) Change working directory
os.chdir("D:/OneDrive - Queen's University/MOMENT/moment-main/data")

# 2) Load the pickle file
pickle_filename = "new_test_set_control_3.pkl"
with open(pickle_filename, "rb") as f:
    new_test_set_control = pickle.load(f)

# 3) Extract data
new_test_Vabs_arr       = new_test_set_control["Vabs"]       # shape (M, 1024)
new_test_Pabs_arr       = new_test_set_control["Pabs"]       # shape (M, 1024)
new_test_Feats_list      = new_test_set_control["Feats"]      # length = M
new_test_Headers_list    = new_test_set_control["Headers"]    # length = M
new_test_Lengths_list    = new_test_set_control["Lengths"]    # length = M (each item is length=16)
new_test_Dir_list = new_test_set_control["Directions"] # length = M (each item is length=16)

# 4) (Optional) Print shapes or a small sample
# print("Vabs shape:", new_test_Vabs_arr.shape)
# print("Pabs shape:", Pabs.shape)
# print("Number of subjects:", len(Headers))
# print("First subject's directions:", Directions[0])










