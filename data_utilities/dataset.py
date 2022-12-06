import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib

from data_utilities import generate_data

matplotlib.use('Agg')


class Dataset(Dataset):
    def __init__(self, num_samples, num_nodes):
        super(Dataset, self).__init__()
        inputs, targets = generate_data.make_seq_data(num_samples, num_nodes)
        self.inputs = torch.LongTensor(inputs)
        self.targets =  torch.LongTensor(targets)
        self.num_samples = num_samples
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):

        return self.inputs[idx], self.targets[idx]
