import numpy as np
import os

class CustomTimeSeriesDataset:
    def __init__(self, data_split="train", channels="both", data_dir="./", forecast_horizon=64, context_len=512):
        """
        Parameters
        ----------
        data_split : str
            Split of the dataset, 'train' or 'test'.
        channels : str
            The channel(s) to use: "Vabs", "Pabs", or "both".
        data_dir : str
            Directory where the .ts files are stored.
        forecast_horizon : int
            Length of the forecast sequence (default is 256).
        context_len : int
            Length of the context window (default is 2048).
        """
        self.forecast_horizon = forecast_horizon
        self.context_len = context_len
        self.channels = channels
        self.data_split = data_split
        self.data_dir = data_dir
        
        # Load data based on split and channels
        self._load_data()

    def _load_ts_data(self, file_path):
        """Load .ts file, skip metadata, and return the numerical time series data."""
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Skip lines until we reach "@data"
        data_start_idx = None
        for i, line in enumerate(lines):
            if "@data" in line:
                data_start_idx = i + 1
                break

        if data_start_idx is None:
            raise ValueError(f"Could not find '@data' in {file_path}")
        
        # Load the numerical data from the file
        data = np.genfromtxt(lines[data_start_idx:], delimiter=',')

        return data

    def _load_data(self):
        """Load data from .ts files based on split and channels."""
        if self.data_split == "train":
            vabs_file = os.path.join(self.data_dir, "vgr_Vabs_TRAIN.ts")
            pabs_file = os.path.join(self.data_dir, "vgr_Pabs_TRAIN.ts")
        elif self.data_split == "test":
            vabs_file = os.path.join(self.data_dir, "vgr_Vabs_TEST.ts")
            pabs_file = os.path.join(self.data_dir, "vgr_Pabs_TEST.ts")
        else:
            raise ValueError("Invalid data_split. Choose 'train' or 'test'.")

        # Load .ts files and extract numerical data
        vabs_data = self._load_ts_data(vabs_file)
        pabs_data = self._load_ts_data(pabs_file)

        # Reshape data into (n_samples, 2304)
        assert vabs_data.shape[1] == 576 and pabs_data.shape[1] == 576, "Unexpected time series length."

        # Select channels based on user input
        if self.channels == "Vabs":
            self.data = vabs_data[:, :576].reshape(vabs_data.shape[0], 1, 576)  # (n_samples, 1, 2304)
        elif self.channels == "Pabs":
            self.data = pabs_data[:, :576].reshape(pabs_data.shape[0], 1, 576)  # (n_samples, 1, 2304)
        elif self.channels == "both":
            self.data = np.stack([vabs_data[:, :576], pabs_data[:, :576]], axis=1)  # (n_samples, 2, 2304)
        else:
            raise ValueError("Invalid channel selection. Choose 'Vabs', 'Pabs', or 'both'.")

        # Print the shape for debugging
        print(f"Loaded data shape: {self.data.shape}")

    def __getitem__(self, index):
        """Return the context and forecast for the selected index."""
        # Context: First 2048 samples
        context = self.data[index, :, :self.context_len]  # (n_channels, context_len)
        
        # Forecast: Last 256 samples
        forecast = self.data[index, :, self.context_len:self.context_len + self.forecast_horizon]  # (n_channels, forecast_len)
        
        # Create a mask of ones for the context (can be used for padding if necessary)
        context_mask = np.ones(self.context_len)

        return context, forecast, context_mask

    def __len__(self):
        """Return the number of samples."""
        return self.data.shape[0]
    
    


# Option 1 (adding directional information as an extra feature channel)
class CustomTimeSeriesDataset_dir1:
    def __init__(self, data_split="train", channels="both", data_dir="./", forecast_horizon=64, context_len=512):
        self.forecast_horizon = forecast_horizon
        self.context_len = context_len
        self.channels = channels
        self.data_split = data_split
        self.data_dir = data_dir

        # Load data and direction files
        self._load_data()
        
    def _load_ts_data(self, file_path):
        """Load .ts file, skip metadata, and return the numerical time series data."""
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Skip lines until we reach "@data"
        data_start_idx = None
        for i, line in enumerate(lines):
            if "@data" in line:
                data_start_idx = i + 1
                break

        if data_start_idx is None:
            raise ValueError(f"Could not find '@data' in {file_path}")
        
        # Load the numerical data from the file
        data = np.genfromtxt(lines[data_start_idx:], delimiter=',')

        return data

    def _load_data(self):
        if self.data_split == "train":
            vabs_file = os.path.join(self.data_dir, "vgr_Vabs_TRAIN.ts")
            pabs_file = os.path.join(self.data_dir, "vgr_Pabs_TRAIN.ts")
            direction_file = os.path.join(self.data_dir, "vgr_Vabs_train_directions.npy")
        elif self.data_split == "test":
            vabs_file = os.path.join(self.data_dir, "vgr_Vabs_TEST.ts")
            pabs_file = os.path.join(self.data_dir, "vgr_Pabs_TEST.ts")
            direction_file = os.path.join(self.data_dir, "vgr_Vabs_test_directions.npy")
        else:
            raise ValueError("Invalid data_split. Choose 'train' or 'test'.")

        vabs_data = self._load_ts_data(vabs_file)
        pabs_data = self._load_ts_data(pabs_file)
        self.directions = np.load(direction_file, allow_pickle=True).astype(int)  # Shape (n_samples, 9)
        
        # Reshape data into (n_samples, 2304)
        assert vabs_data.shape[1] == 576 and pabs_data.shape[1] == 576, "Unexpected time series length."
        
        # Assert that vabs_data and direction labels match in length
        assert vabs_data.shape[0] == self.directions.shape[0], "Mismatch in data and direction lengths."

        # Select channels based on user input
        if self.channels == "Vabs":
            self.data = vabs_data[:, :576].reshape(vabs_data.shape[0], 1, 576)  # (n_samples, 1, 2304)
        elif self.channels == "Pabs":
            self.data = pabs_data[:, :576].reshape(pabs_data.shape[0], 1, 576)  # (n_samples, 1, 2304)
        elif self.channels == "both":
            self.data = np.stack([vabs_data[:, :576], pabs_data[:, :576]], axis=1)  # (n_samples, 2, 2304)
        else:
            raise ValueError("Invalid channel selection. Choose 'Vabs', 'Pabs', or 'both'.")

    def __getitem__(self, index):
        # Extract the 512-sample context (8 trials with directions)
        context = self.data[index, :, :self.context_len]  # Shape: (1, 512)
        context_directions = self.directions[index, :8]  # First 8 directions
        
        # Extract the forecast trial and its direction
        forecast = self.data[index, :, self.context_len:self.context_len + self.forecast_horizon]  # Shape: (1, 64)
        forecast_direction = self.directions[index, 8]  # Direction of the forecast trial

        # Expand context directions to match the context length (512 samples)
        context_directions_expanded = np.repeat(context_directions, 64)  # Shape: (512,)
        context_directions_expanded = context_directions_expanded[np.newaxis, :]  # Shape: (1, 512)

        # Add directions as an additional feature channel
        context = np.concatenate([context, context_directions_expanded], axis=0)  # Shape: (2, 512)

        # Add the forecast direction as a constant channel
        forecast_direction_channel = np.full((1, 64), forecast_direction)  # Shape: (1, 64)
        forecast = np.concatenate([forecast, forecast_direction_channel], axis=0)  # Shape: (2, 64)

        context_mask = np.ones(self.context_len)

        return context, forecast, context_mask, forecast_direction

    def __len__(self):
        return self.data.shape[0]
    
    
    
    
    
# Option 2: To implement "Encoding Direction as an Embedding", we will use a learnable embedding layer for directions, replacing the current approach where direction is concatenated as a feature channel.
class CustomTimeSeriesDataset_dir2:
    def __init__(self, data_split="train", channels="both", data_dir="./", forecast_horizon=64, context_len=512):
        self.forecast_horizon = forecast_horizon
        self.context_len = context_len
        self.channels = channels
        self.data_split = data_split
        self.data_dir = data_dir

        # Load data and direction files
        self._load_data()
        
    def _load_ts_data(self, file_path):
        """Load .ts file, skip metadata, and return the numerical time series data."""
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Skip lines until we reach "@data"
        data_start_idx = None
        for i, line in enumerate(lines):
            if "@data" in line:
                data_start_idx = i + 1
                break

        if data_start_idx is None:
            raise ValueError(f"Could not find '@data' in {file_path}")
        
        # Load the numerical data from the file
        data = np.genfromtxt(lines[data_start_idx:], delimiter=',')

        return data

    def _load_data(self):
        if self.data_split == "train":
            vabs_file = os.path.join(self.data_dir, "vgr_Vabs_TRAIN.ts")
            pabs_file = os.path.join(self.data_dir, "vgr_Pabs_TRAIN.ts")
            direction_file = os.path.join(self.data_dir, "vgr_Vabs_train_directions.npy")
        elif self.data_split == "test":
            vabs_file = os.path.join(self.data_dir, "vgr_Vabs_TEST.ts")
            pabs_file = os.path.join(self.data_dir, "vgr_Pabs_TEST.ts")
            direction_file = os.path.join(self.data_dir, "vgr_Vabs_test_directions.npy")
        else:
            raise ValueError("Invalid data_split. Choose 'train' or 'test'.")

        vabs_data = self._load_ts_data(vabs_file)
        pabs_data = self._load_ts_data(pabs_file)
        self.directions = np.load(direction_file, allow_pickle=True).astype(int)  # Shape (n_samples, 9)
        
        # Reshape data into (n_samples, 2304)
        assert vabs_data.shape[1] == 576 and pabs_data.shape[1] == 576, "Unexpected time series length."
        
        # Assert that vabs_data and direction labels match in length
        assert vabs_data.shape[0] == self.directions.shape[0], "Mismatch in data and direction lengths."

        # Select channels based on user input
        if self.channels == "Vabs":
            self.data = vabs_data[:, :576].reshape(vabs_data.shape[0], 1, 576)  # (n_samples, 1, 2304)
        elif self.channels == "Pabs":
            self.data = pabs_data[:, :576].reshape(pabs_data.shape[0], 1, 576)  # (n_samples, 1, 2304)
        elif self.channels == "both":
            self.data = np.stack([vabs_data[:, :576], pabs_data[:, :576]], axis=1)  # (n_samples, 2, 2304)
        else:
            raise ValueError("Invalid channel selection. Choose 'Vabs', 'Pabs', or 'both'.")

    def __getitem__(self, index):
        # Extract the 512-sample context (8 trials with directions)
        context = self.data[index, :, :self.context_len]  # Shape: (n_channels, 512)
        context_directions = self.directions[index, :8] - 2  # First 8 directions as embedding indices (range 0-7)
    
        # Extract the forecast trial and its direction
        forecast = self.data[index, :, self.context_len:self.context_len + self.forecast_horizon]  # Shape: (n_channels, 64)
        forecast_direction = self.directions[index, 8] - 2  # Forecast direction as embedding index
    
        context_mask = np.ones(self.context_len)
    
        return context, forecast, context_mask, context_directions, forecast_direction
    
    def __len__(self):
        return self.data.shape[0]
    
###########################################################loss_related
# data_lengths.py or at the top of your dataset script
import pickle
import numpy as np

def load_data_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

# -- Load everything for both groups (control + stroke)
control_Vabs_data   = load_data_pickle('./data/control_Vabs_data.pkl')
control_headers     = load_data_pickle('./data/control_headers.pkl')
control_lengths     = load_data_pickle('./data/control_lengths.pkl')
stroke_Vabs_data    = load_data_pickle('./data/stroke_Vabs_data.pkl')
stroke_headers      = load_data_pickle('./data/stroke_headers.pkl')
stroke_lengths      = load_data_pickle('./data/stroke_lengths.pkl')

# Combine
combined_headers = control_headers + stroke_headers
combined_data    = control_Vabs_data + stroke_Vabs_data
combined_lengths = control_lengths + stroke_lengths

# e.g.  *One* function for cross matching:
# file: data_lengths.py  (or wherever you define cross-matching)

def find_matching_samples_9(query_ids, all_headers, all_lengths):
    """
    query_ids: array/list of length N, each representing a dataset sample ID.
    all_headers: the combined headers for all subjects, each a list with a unique ID at header[-1].
    all_lengths: the corresponding list of lists/arrays of lengths for each subject/entry.

    Returns: a list of shape (N, 9), i.e. each element is a list of 9 floats.
    """
    matched_2d = []

    for q_id in query_ids:
        found_9 = None
        # search in all_headers + all_lengths
        for header, length_arr in zip(all_headers, all_lengths):
            if q_id == header[-1]:
                # We assume 'length_arr' might contain length info for multiple trials (>=9).
                # We only take the first 9:
                found_9 = list(length_arr[:9])
                break

        # If not found, or length_arr had fewer than 9, pad
        if found_9 is None:
            found_9 = [1000.0]*9
        elif len(found_9) < 9:
            # pad
            needed = 9 - len(found_9)
            found_9 += [1000.0]*needed

        matched_2d.append(found_9)

    # matched_2d is shape (N, 9). Each row is a 9-element list.
    return matched_2d


#############################################################    
    
    
# Option 3: To implement Direction-Aware Positional Encoding
# class CustomTimeSeriesDataset_dir3:
#     def __init__(self, data_split="train", channels="both", data_dir="./", forecast_horizon=64, context_len=512):
#         self.forecast_horizon = forecast_horizon
#         self.context_len = context_len
#         self.channels = channels
#         self.data_split = data_split
#         self.data_dir = data_dir

#         # Load data and direction files
#         self._load_data()
        
#     def _load_ts_data(self, file_path):
#         """Load .ts file, skip metadata, and return the numerical time series data."""
#         with open(file_path, 'r') as f:
#             lines = f.readlines()

#         # Skip lines until we reach "@data"
#         data_start_idx = None
#         for i, line in enumerate(lines):
#             if "@data" in line:
#                 data_start_idx = i + 1
#                 break

#         if data_start_idx is None:
#             raise ValueError(f"Could not find '@data' in {file_path}")
        
#         # Load the numerical data from the file
#         data = np.genfromtxt(lines[data_start_idx:], delimiter=',')

#         return data

#     def _load_data(self):
#         if self.data_split == "train":
#             vabs_file = os.path.join(self.data_dir, "vgr_Vabs_TRAIN.ts")
#             pabs_file = os.path.join(self.data_dir, "vgr_Pabs_TRAIN.ts")
#             direction_file = os.path.join(self.data_dir, "vgr_Vabs_train_directions.npy")
#         elif self.data_split == "test":
#             vabs_file = os.path.join(self.data_dir, "vgr_Vabs_TEST.ts")
#             pabs_file = os.path.join(self.data_dir, "vgr_Pabs_TEST.ts")
#             direction_file = os.path.join(self.data_dir, "vgr_Vabs_test_directions.npy")
#         else:
#             raise ValueError("Invalid data_split. Choose 'train' or 'test'.")

#         vabs_data = self._load_ts_data(vabs_file)
#         pabs_data = self._load_ts_data(pabs_file)
#         self.directions = np.load(direction_file, allow_pickle=True).astype(int)  # Shape (n_samples, 9)
        
#         # Reshape data into (n_samples, 2304)
#         assert vabs_data.shape[1] == 576 and pabs_data.shape[1] == 576, "Unexpected time series length."
        
#         # Assert that vabs_data and direction labels match in length
#         assert vabs_data.shape[0] == self.directions.shape[0], "Mismatch in data and direction lengths."

#         # Select channels based on user input
#         if self.channels == "Vabs":
#             self.data = vabs_data[:, :576].reshape(vabs_data.shape[0], 1, 576)  # (n_samples, 1, 2304)
#         elif self.channels == "Pabs":
#             self.data = pabs_data[:, :576].reshape(pabs_data.shape[0], 1, 576)  # (n_samples, 1, 2304)
#         elif self.channels == "both":
#             self.data = np.stack([vabs_data[:, :576], pabs_data[:, :576]], axis=1)  # (n_samples, 2, 2304)
#         else:
#             raise ValueError("Invalid channel selection. Choose 'Vabs', 'Pabs', or 'both'.")

#     # def __getitem__(self, index):
#     #     # Extract the 512-sample context (8 trials with directions)
#     #     context = self.data[index, :, :self.context_len]  # Shape: (n_channels, 512)
#     #     context_directions = self.directions[index, :8]  # Shape: (8,)
#     #     forecast_direction = self.directions[index, 8] - 2  # Forecast direction as embedding index

#     #     # Extract the forecast trial
#     #     forecast = self.data[index, :, self.context_len:self.context_len + self.forecast_horizon]  # Shape: (n_channels, 64)
#     #     context_mask = np.ones(self.context_len)

#     #     return context, forecast, context_mask, context_directions, forecast_direction
    
#     def __getitem__(self, index):
#         # Extract the 512-sample context (8 trials with directions)
#         context = self.data[index, :, :self.context_len]  # Shape: (n_channels, 512)
    
#         # Original directions for the 8 trials
#         original_directions = self.directions[index, :8]  # Shape: (8,)
    
#         # Repeat each direction 8 times to match 64 patches
#         replicated_directions = []
#         for d in original_directions:
#             replicated_directions.extend([d - 2] * 8)  # Subtract 2 if embedding indices start at 0
    
#         context_directions = np.array(replicated_directions, dtype=int)  # Shape: (64,)
    
#         # Forecast direction
#         forecast_direction = self.directions[index, 8] - 2  # Single direction index for the forecast
    
#         # Extract the forecast trial
#         forecast = self.data[index, :, self.context_len : self.context_len + self.forecast_horizon]  # (n_channels, 64)
#         context_mask = np.ones(self.context_len)
    
#         return context, forecast, context_mask, context_directions, forecast_direction

    
#     def __len__(self):
#         return self.data.shape[0]




# class CustomTimeSeriesDataset_dir3:
#     def __init__(self, data_split="train", channels="both", data_dir="./", forecast_horizon=64, context_len=512):
#         self.forecast_horizon = forecast_horizon
#         self.context_len = context_len
#         self.channels = channels
#         self.data_split = data_split
#         self.data_dir = data_dir

#         # 2) Load .ts data, directions, etc.
#         self._load_data()

#         # 3) Load the corresponding headers so we can cross-match
#         if self.data_split == "train":
#             header_file = os.path.join(self.data_dir, "vgr_Vabs_train_headers.npy")
#         elif self.data_split == "test":
#             header_file = os.path.join(self.data_dir, "vgr_Vabs_test_headers.npy")
#         else:
#             raise ValueError("Invalid data_split. Choose 'train' or 'test'.")

#         # shape = (n_samples, some_number_of_cols)
#         self.headers_array = np.load(header_file, allow_pickle=True)
#         # The last column is the unique ID
#         self.sample_ids = self.headers_array[:, -1]  # shape (n_samples,)

#         # 4) cross-match => returns list of shape (n_samples, 9)
#         matched_lengths_2d = find_matching_samples_9(
#             query_ids=self.sample_ids,
#             all_headers=combined_headers,
#             all_lengths=combined_lengths
#         )

#         # 5) Convert to float32 array => shape (n_samples, 9)
#         self.original_lengths = np.array(matched_lengths_2d, dtype=np.float32)

#     def _load_ts_data(self, file_path):
#         with open(file_path, 'r') as f:
#             lines = f.readlines()
#         data_start_idx = None
#         for i, line in enumerate(lines):
#             if "@data" in line:
#                 data_start_idx = i + 1
#                 break
#         if data_start_idx is None:
#             raise ValueError(f"Could not find '@data' in {file_path}")
#         data = np.genfromtxt(lines[data_start_idx:], delimiter=',')
#         return data

#     def _load_data(self):
#         if self.data_split == "train":
#             vabs_file = os.path.join(self.data_dir, "vgr_Vabs_TRAIN.ts")
#             pabs_file = os.path.join(self.data_dir, "vgr_Pabs_TRAIN.ts")
#             direction_file = os.path.join(self.data_dir, "vgr_Vabs_train_directions.npy")
#         elif self.data_split == "test":
#             vabs_file = os.path.join(self.data_dir, "vgr_Vabs_TEST.ts")
#             pabs_file = os.path.join(self.data_dir, "vgr_Pabs_TEST.ts")
#             direction_file = os.path.join(self.data_dir, "vgr_Vabs_test_directions.npy")
#         else:
#             raise ValueError("Invalid data_split. Choose 'train' or 'test'.")

#         vabs_data = self._load_ts_data(vabs_file)
#         pabs_data = self._load_ts_data(pabs_file)
#         self.directions = np.load(direction_file, allow_pickle=True).astype(int)
        
#         if (vabs_data.shape[1] != 576) or (pabs_data.shape[1] != 576):
#             raise ValueError("Unexpected time series length; expected 576 each.")

#         if vabs_data.shape[0] != self.directions.shape[0]:
#             raise ValueError("Mismatch in data and direction lengths.")

#         # 6) select channels
#         if self.channels == "Vabs":
#             self.data = vabs_data[:, :576].reshape(vabs_data.shape[0], 1, 576)
#         elif self.channels == "Pabs":
#             self.data = pabs_data[:, :576].reshape(pabs_data.shape[0], 1, 576)
#         elif self.channels == "both":
#             self.data = np.stack([vabs_data[:, :576], pabs_data[:, :576]], axis=1)  # (n_samples,2,576)
#         else:
#             raise ValueError("Invalid channel selection.")

#     def __getitem__(self, index):
#         # 7) Build the standard (context, forecast, directions, etc.)
#         context = self.data[index, :, :self.context_len]  # shape (n_channels, 512)
#         original_directions = self.directions[index, :8]  # shape (8,)
#         # replicate to shape (64,)  
#         replicated_directions = []
#         for d in original_directions:
#             replicated_directions.extend([d - 2]*8)
#         context_directions = np.array(replicated_directions, dtype=int)  # (64,)

#         forecast_direction = self.directions[index, 8] - 2
#         forecast = self.data[index, :, self.context_len : self.context_len + self.forecast_horizon]
#         context_mask = np.ones(self.context_len, dtype=np.float32)

#         # 8) Retrieve 9 lengths => shape (9,)
#         lengths_9 = self.original_lengths[index]  # e.g. [l0, l1, ..., l8]

#         # If you only need the forecast trial's length:
#         forecast_length = lengths_9[8]
#         # If you also want the 8 context trial lengths:
#         context_lengths = lengths_9[:8]

#         return (context,
#                 forecast,
#                 context_mask,
#                 context_directions,
#                 forecast_direction,
#                 context_lengths,
#                 forecast_length)

#     def __len__(self):
#         return self.data.shape[0]


import torch
import torch.nn.functional as F

class CustomTimeSeriesDataset_dir3_set:
    def __init__(self, data_split="train", channels="both", data_dir="./", forecast_horizon=64, context_len=512):
        self.forecast_horizon = forecast_horizon
        self.context_len = context_len
        self.channels = channels
        self.data_split = data_split
        self.data_dir = data_dir

        # 2) Load .ts data, directions, etc.
        self._load_data()

        # 3) Load the corresponding headers so we can cross-match
        if self.data_split == "train":
            header_file = os.path.join(self.data_dir, "vgr_Vabs_train_headers.npy")
        elif self.data_split == "test":
            header_file = os.path.join(self.data_dir, "vgr_Vabs_test_headers.npy")
        else:
            raise ValueError("Invalid data_split. Choose 'train' or 'test'.")

        # shape = (n_samples, some_number_of_cols)
        self.headers_array = np.load(header_file, allow_pickle=True)
        # The last column is the unique ID
        self.sample_ids = self.headers_array[:, -1]  # shape (n_samples,)

        # 4) cross-match => returns list of shape (n_samples, 9)
        matched_lengths_2d = find_matching_samples_9(
            query_ids=self.sample_ids,
            all_headers=combined_headers,
            all_lengths=combined_lengths
        )

        # 5) Convert to float32 array => shape (n_samples, 9)
        self.original_lengths = np.array(matched_lengths_2d, dtype=np.float32)

    def _load_ts_data(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        data_start_idx = None
        for i, line in enumerate(lines):
            if "@data" in line:
                data_start_idx = i + 1
                break
        if data_start_idx is None:
            raise ValueError(f"Could not find '@data' in {file_path}")
        data = np.genfromtxt(lines[data_start_idx:], delimiter=',')
        return data

    def _load_data(self):
        if self.data_split == "train":
            vabs_file = os.path.join(self.data_dir, "vgr_Vabs_TRAIN.ts")
            pabs_file = os.path.join(self.data_dir, "vgr_Pabs_TRAIN.ts")
            direction_file = os.path.join(self.data_dir, "vgr_Vabs_train_directions.npy")
        elif self.data_split == "test":
            vabs_file = os.path.join(self.data_dir, "vgr_Vabs_TEST.ts")
            pabs_file = os.path.join(self.data_dir, "vgr_Pabs_TEST.ts")
            direction_file = os.path.join(self.data_dir, "vgr_Vabs_test_directions.npy")
        else:
            raise ValueError("Invalid data_split. Choose 'train' or 'test'.")

        vabs_data = self._load_ts_data(vabs_file)
        pabs_data = self._load_ts_data(pabs_file)
        self.directions = np.load(direction_file, allow_pickle=True).astype(int)
        
        if (vabs_data.shape[1] != 576) or (pabs_data.shape[1] != 576):
            raise ValueError("Unexpected time series length; expected 576 each.")

        if vabs_data.shape[0] != self.directions.shape[0]:
            raise ValueError("Mismatch in data and direction lengths.")

        # 6) select channels
        if self.channels == "Vabs":
            self.data = vabs_data[:, :576].reshape(vabs_data.shape[0], 1, 576)
        elif self.channels == "Pabs":
            self.data = pabs_data[:, :576].reshape(pabs_data.shape[0], 1, 576)
        elif self.channels == "both":
            self.data = np.stack([vabs_data[:, :576], pabs_data[:, :576]], axis=1)  # (n_samples,2,576)
        else:
            raise ValueError("Invalid channel selection.")

    def __getitem__(self, index):
        # n_channels, total_len=576
        n_channels = self.data[index].shape[0]

        # 1) Raw context => shape (n_channels, 512) => 8 trials x 64 each
        raw_context = self.data[index, :, :self.context_len]  # (n_channels, 512)
        # Reshape => (n_channels, 8, 64)
        raw_trials = raw_context.reshape(n_channels, 8, 64)

        # 2) Upsample each trial from 64 -> 512
        raw_trials_tensor = torch.tensor(raw_trials, dtype=torch.float32)  # (n_channels, 8, 64)
        c_nc, c_ntrials, c_len = raw_trials_tensor.shape  # c_ntrials=8, c_len=64
        reshaped = raw_trials_tensor.reshape(c_nc*c_ntrials, 1, c_len)  # => (n_channels*8, 1, 64)
        upsampled = F.interpolate(reshaped, size=512, mode='linear', align_corners=False)
        # => (n_channels*8, 1, 512)
        upsampled = upsampled.reshape(c_nc, c_ntrials, 512)  # => (n_channels, 8, 512)
        # concat => (n_channels, 4096)
        context = upsampled.reshape(c_nc, -1)

        # 3) context directions => shape (8,)
        context_dirs = self.directions[index, :8] - 2
        forecast_dir = self.directions[index, 8] - 2

        # 4) Forecast trial => shape (n_channels, 64) => upsample 64->512
        raw_forecast = self.data[index, :, self.context_len : self.context_len + self.forecast_horizon]  # => (n_channels, 64)
        raw_forecast_tensor = torch.tensor(raw_forecast, dtype=torch.float32).unsqueeze(1)  # => (n_channels,1,64)
        upsampled_forecast = F.interpolate(raw_forecast_tensor, size=512, mode='linear', align_corners=False)
        forecast = upsampled_forecast.squeeze(1)  # => (n_channels, 512)

        # 5) context mask => 4096
        context_mask = np.ones(context.shape[1], dtype=np.float32)  # => shape (4096,)

        # If needed, context_lengths & forecast_length = 512
        context_lengths = [512]*8
        forecast_length = 512
        
        replicated_dirs = []
        for d in context_dirs:
            replicated_dirs.extend([d]*64)
        context_dirs_512 = np.array(replicated_dirs, dtype=int)  # shape (512,)

        return (
            context,            # shape (n_channels, 4096)
            forecast,           # shape (n_channels, 512)
            context_mask,       # shape (4096,)
            context_dirs,       # shape (8,)
            forecast_dir,       # scalar
            context_lengths,    # [512, ..., 512]
            forecast_length     # 512
        )


    def __len__(self):
        return self.data.shape[0]
    
    
    
    
    
    
# singletrial_dataset.py


from torch.utils.data import Dataset

class CustomTimeSeriesDataset_dir3_SingleTrial(Dataset):
    """
    Returns single trials (64 samples) from a subject's total 576-sample record.
    Each subject has 9 trials => 8 context, 1 forecast.
    direction[i] => direction index for trial i
    """

    def __init__(self, data_split="train", channels="Vabs", data_dir="./"):
        super().__init__()
        self.data_split = data_split
        self.channels = channels
        self.data_dir = data_dir
        self.horizon = 64  # each trial is 64

        self._load_data()  # => self.data: shape [n_subjects, n_channels, 576], self.directions: shape [n_subjects,9]

        self.index_list = []
        n_subjects = self.data.shape[0]
        for subj_id in range(n_subjects):
            for trial_idx in range(9):
                self.index_list.append((subj_id, trial_idx))

    def _load_ts_data(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        data_start_idx = None
        for i, line in enumerate(lines):
            if "@data" in line:
                data_start_idx = i+1
                break
        if data_start_idx is None:
            raise ValueError("No '@data' found in file " + file_path)
        arr = np.genfromtxt(lines[data_start_idx:], delimiter=',')
        return arr

    def _load_data(self):
        # example: handle only Vabs for brevity
        if self.data_split == "train":
            vabs_file = os.path.join(self.data_dir, "vgr_Vabs_TRAIN.ts")
            direction_file = os.path.join(self.data_dir, "vgr_Vabs_train_directions.npy")
        else:
            vabs_file = os.path.join(self.data_dir, "vgr_Vabs_TEST.ts")
            direction_file = os.path.join(self.data_dir, "vgr_Vabs_test_directions.npy")

        vabs_data = self._load_ts_data(vabs_file)  # => shape [n_subjects,576]
        self.directions = np.load(direction_file, allow_pickle=True).astype(int) # => shape [n_subjects,9]

        # reshape => [n_subjects,1,576]
        self.data = vabs_data.reshape(vabs_data.shape[0], 1, 576)

    def __getitem__(self, idx):
        subj_id, trial_idx = self.index_list[idx]

        start = trial_idx * 64
        end   = start + 64

        # shape => (n_channels,64)
        trial_data = self.data[subj_id, :, start:end]
        direction  = self.directions[subj_id, trial_idx] - 2  # e.g. shift index
        is_forecast = (trial_idx == 8)

        return {
           "subject_id": subj_id,
           "trial_data": trial_data,     # (n_channels,64)
           "direction": direction,
           "is_forecast": is_forecast
        }

    def __len__(self):
        return len(self.index_list)















