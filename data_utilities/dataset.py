import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib

from data_utilities import generate_data

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Dataset(Dataset):
    def __init__(self, num_samples, nodes_num):
        super(Dataset, self).__init__()
        inputs, targets = generate_data.make_seq_data(num_samples, nodes_num)

    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
