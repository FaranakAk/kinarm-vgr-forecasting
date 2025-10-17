# -*- coding: utf-8 -*-
"""
Created on Mon May  5 14:45:31 2025

@author: fakbarifar
"""
import os
os.chdir("D:/OneDrive - Queen's University/MOMENT/moment-main")
import h5py
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from scipy.signal import resample

from gluonts.dataset.arrow import ArrowWriter
import pandas as pd
from pathlib import Path

##############################################################################
# 1) Parameters: "control", "stroke", or "both"
##############################################################################
GROUP_TO_PROCESS = "both"  # options: "control", "stroke", "both"

CONTROL_INPUT_FILENAME = 'selected_control_raw_VandP_Pcorrected_cropped_Mar_2025_adv.h5'
STROKE_INPUT_FILENAME = 'selected_stroke_raw_VandP_Pcorrected_cropped_Mar_2025_adv.h5'



OUTPUT_DIR = "vgr_raw_arrow"
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





def write_arrow(target_array, out_path):
    """
    target_array: numpy array of shape (N_subjects, 576)
    out_path    : Path('train.arrow') or 'test.arrow'
    """
    rows = []
    dummy_start = pd.Timestamp("2000-01-01 00:00")   # any timestamp is fine
    for series in target_array:
        rows.append({"start": dummy_start, "target": series.astype(np.float32)})
    ArrowWriter(compression="lz4").write_to_file(rows, Path(out_path))




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




#10
# ---- write files ----
write_arrow(Vabs_train_data, os.path.join(OUTPUT_DIR,"vgr_Vabs_train.arrow"))
write_arrow(Vabs_test_data,  os.path.join(OUTPUT_DIR,"vgr_Vabs_test.arrow"))   # for your final evaluation

write_arrow(Pabs_train_data, os.path.join(OUTPUT_DIR,"vgr_Pabs_train.arrow"))
write_arrow(Pabs_test_data,  os.path.join(OUTPUT_DIR,"vgr_Pabs_test.arrow"))   # for your final evaluation

print(f"Done. Created train/test split at subject level for {GROUP_TO_PROCESS} data.")