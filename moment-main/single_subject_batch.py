# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:20:20 2025

@author: fakbarifar
"""
# single_subject_batch.py
import numpy as np
import torch
from torch.utils.data import Sampler

class SubjectBatchSampler(Sampler):
    """
    Yields consecutive groups of 9 single-trial items for each subject.
    We assume dataset length= n_subjects * 9, each subject has 9 trials.
    """
    def __init__(self, data_source, trials_per_subject=9):
        super().__init__(data_source)
        self.data_source = data_source
        self.trials_per_subject = trials_per_subject
        self.n_items = len(data_source)

    def __iter__(self):
        idxs = np.arange(self.n_items)
        for i in range(0, self.n_items, self.trials_per_subject):
            yield idxs[i:i+self.trials_per_subject].tolist()

    def __len__(self):
        return self.n_items // self.trials_per_subject

def collate_fn_single_subject(batch_list):
    """
    batch_list: 9 single-trial items from the same subject
    => separate 8 context vs 1 forecast
    => produce shapes for pipeline:
       context => (n_channels,8,64)
       context_directions => (8,)
       forecast => (n_channels,64)
       forecast_direction => scalar
       input_mask => (8,64)
    """
    context_list = []
    context_dirs = []
    forecast_trial = None
    forecast_dir   = None
    n_channels     = None
    
    # print('************************', len(batch_list))
    # for i, item in enumerate(batch_list):
    #     print(f"--- item {i} subject={item['subject_id']} trial_idx=?? is_forecast={item['is_forecast']}, direction={item['direction']}")

    for item in batch_list:
        if n_channels is None:
            n_channels = item["trial_data"].shape[0]

        if item["is_forecast"]:
            forecast_trial = item["trial_data"]  # (n_channels,64)
            forecast_dir   = item["direction"]
        else:
            context_list.append(item["trial_data"])
            context_dirs.append(item["direction"])
            
    # # Debug printing
    # print("DEBUG: In collate_fn => #context_trials=", len(context_list), len(forecast_trial),
    #       " forecast_trial_is_none=", (forecast_trial is None))

    # stack => shape (8, n_channels,64)
    context_tensor = torch.stack([torch.tensor(x, dtype=torch.float32) for x in context_list], dim=0)
    # => (8, n_channels,64)
    # transpose => (n_channels,8,64)
    context_tensor = context_tensor.transpose(0,1)

    forecast_tensor = torch.tensor(forecast_trial, dtype=torch.float32) # => (n_channels,64)

    context_dirs = torch.tensor(context_dirs, dtype=torch.long) # => (8,)
    forecast_dir = torch.tensor([forecast_dir], dtype=torch.long)[0]

    input_mask = torch.ones((8,64), dtype=torch.float32)

    # We'll return placeholders for context_lengths, forecast_length if code checks them:
    return (
       context_tensor,           # (n_channels,8,64)
       forecast_tensor,          # (n_channels,64)
       input_mask,               # (8,64)
       context_dirs,             # (8,)
       forecast_dir,             # scalar
       None,                     # context_lengths
       None                      # forecast_length
    )
