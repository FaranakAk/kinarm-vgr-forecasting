import numpy as np
import os
from sklearn.preprocessing import StandardScaler

from momentfm.utils.data import load_from_tsfile


class ClassificationDataset:
    def __init__(self, data_split="train"):
        """
        Parameters
        ----------
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        """

        self.seq_len = 576
        # Get the current file directory
        current_dir = os.getcwd()
        
        # ECG data
        # self.train_file_path_and_name = os.path.join(current_dir, "data", "ECG5000_TRAIN.ts")
        # self.test_file_path_and_name = os.path.join(current_dir, "data", "ECG5000_TEST.ts")
        
        # vgr_Vabs data
        self.train_file_path_and_name = os.path.join(current_dir, "data", "vgr_Vabs_TRAIN.ts")
        self.test_file_path_and_name = os.path.join(current_dir, "data", "vgr_Vabs_TEST.ts")
        
        # self.train_file_path_and_name = os.path.join(current_dir, "data", "vgr_Pabs_TRAIN.ts")
        # self.test_file_path_and_name = os.path.join(current_dir, "data", "vgr_Pabs_TEST.ts")
        
        self.data_split = data_split  # 'train' or 'test'

        # Read data
        self._read_data()

    def _transform_labels(self, train_labels: np.ndarray, test_labels: np.ndarray):
        labels = np.unique(train_labels)  # Move the labels to {0, ..., L-1}
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&", train_labels)
        transform = {}
        for i, l in enumerate(labels):
            transform[l] = i

        train_labels = np.vectorize(transform.get)(train_labels)
        test_labels = np.vectorize(transform.get)(test_labels)

        return train_labels, test_labels

    def __len__(self):
        return self.num_timeseries

    def _read_data(self):
        self.scaler = StandardScaler()

        self.train_data, self.train_labels = load_from_tsfile(
            self.train_file_path_and_name
        )
        self.test_data, self.test_labels = load_from_tsfile(
            self.test_file_path_and_name
        )

        self.train_labels, self.test_labels = self._transform_labels(
            self.train_labels, self.test_labels
        )

        if self.data_split == "train":
            self.data = self.train_data
            self.labels = self.train_labels
        else:
            self.data = self.test_data
            self.labels = self.test_labels

        self.num_timeseries = self.data.shape[0]
        self.len_timeseries = self.data.shape[2]

        self.data = self.data.reshape(-1, self.len_timeseries)
        self.scaler.fit(self.data)
        self.data = self.scaler.transform(self.data)
        self.data = self.data.reshape(self.num_timeseries, self.len_timeseries)

        self.data = self.data.T

    def __getitem__(self, index):
        assert index < self.__len__()

        timeseries = self.data[:, index]
        timeseries_len = len(timeseries)
        labels = self.labels[index,].astype(int)
        input_mask = np.ones(self.seq_len)
        input_mask[: self.seq_len - timeseries_len] = 0

        timeseries = np.pad(timeseries, (self.seq_len - timeseries_len, 0))

        return np.expand_dims(timeseries, axis=0), input_mask, labels







class ClassificationDataset_multi:
    def __init__(self, data_split="train"):
        """
        Parameters
        ----------
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        """

        self.seq_len = 576
        # Get the current file directory
        current_dir = os.getcwd()
        
        # Paths for Vabs and Pabs data
        self.vabs_train_file = os.path.join(current_dir, "data", "vgr_Vabs_TRAIN.ts")
        self.vabs_test_file = os.path.join(current_dir, "data", "vgr_Vabs_TEST.ts")
        self.pabs_train_file = os.path.join(current_dir, "data", "vgr_Pabs_TRAIN.ts")
        self.pabs_test_file = os.path.join(current_dir, "data", "vgr_Pabs_TEST.ts")
        
        self.data_split = data_split  # 'train' or 'test'

        # Read and combine data
        self._read_data()

    def _transform_labels(self, vabs_labels: np.ndarray, pabs_labels: np.ndarray):
        # Assuming both Vabs and Pabs have the same labels
        labels = np.unique(vabs_labels)  
        transform = {}
        for i, l in enumerate(labels):
            transform[l] = i

        vabs_labels = np.vectorize(transform.get)(vabs_labels)
        pabs_labels = np.vectorize(transform.get)(pabs_labels)

        return vabs_labels, pabs_labels

    def __len__(self):
        return self.num_timeseries

    def _read_data(self):
        self.scaler = StandardScaler()

        if self.data_split == "train":
            vabs_data, vabs_labels = load_from_tsfile(self.vabs_train_file)
            pabs_data, pabs_labels = load_from_tsfile(self.pabs_train_file)
        else:
            vabs_data, vabs_labels = load_from_tsfile(self.vabs_test_file)
            pabs_data, pabs_labels = load_from_tsfile(self.pabs_test_file)

        vabs_labels, pabs_labels = self._transform_labels(vabs_labels, pabs_labels)

        # Ensure the labels are the same for both datasets
        assert np.array_equal(vabs_labels, pabs_labels), "Vabs and Pabs labels must match."

        # Combine Vabs and Pabs data into two-channel data
        vabs_data_squeezed = np.squeeze(vabs_data, axis=1)  # Shape: (861, 2304)
        pabs_data_squeezed = np.squeeze(pabs_data, axis=1)  # Shape: (861, 2304)
        
        combined_data = np.stack([vabs_data_squeezed, pabs_data_squeezed], axis=1)  # Shape: (861, 2, 2304)

        self.data = combined_data
        self.labels = vabs_labels  # Or pabs_labels since they are the same

        self.num_timeseries = self.data.shape[0]
        self.len_timeseries = self.data.shape[2]

        # Reshape and scale each channel independently
        for i in range(2):  # For each channel
            channel_data = self.data[:, i, :].reshape(-1, self.len_timeseries)
            self.scaler.fit(channel_data)
            scaled_data = self.scaler.transform(channel_data)
            self.data[:, i, :] = scaled_data.reshape(self.num_timeseries, self.len_timeseries)

    def __getitem__(self, index):
        assert index < self.__len__()

        timeseries = self.data[index, :, :]  # Get both channels
        timeseries_len = timeseries.shape[1]
        labels = self.labels[index].astype(int)
        input_mask = np.ones(self.seq_len)
        input_mask[: self.seq_len - timeseries_len] = 0

        # Padding for each channel
        padded_timeseries = np.zeros((2, self.seq_len))
        padded_timeseries[:, self.seq_len - timeseries_len:] = timeseries

        return padded_timeseries, input_mask, labels
