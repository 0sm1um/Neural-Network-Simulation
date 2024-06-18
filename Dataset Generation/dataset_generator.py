import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np


# Datasets should be specified on a per simulation basis

#class sequence_generator(Dataset):

class dataset_composer(Dataset):
    #Range Bearing 2D Dataset Sequence Packer

    def __init__(self, dataset):
        self.dataset = dataset
    
    def append_to_dataset(self, ground_truth, measurements):
        """
        Parameters
        ----------
        ground_truth : :class:`~.list`
            A list of Stonesoup States.
        measurements : :class:`~.measurements`
            The list of Stonesoup Measurements to be used.
        """
    
        gt = np.array([e.state_vector for e in ground_truth]).squeeze().T
        gt = torch.tensor(gt.astype(dtype=np.float64))
        ms = np.array([m.state_vector for m in measurements]).squeeze().T
        ms = torch.tensor(ms.astype(dtype=np.float64))
        new_entry = torch.cat((gt,ms))

        dataset = torch.cat((dataset,torch.unsqueeze(new_entry,dim=0)),dim=0)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return self.dataset.shape[0]

