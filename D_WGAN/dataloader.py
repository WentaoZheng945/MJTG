import numpy as np
from torch.utils.data import Dataset
import random
import os


class Sequence_Dataset(Dataset):
    def __init__(self, data_path):
        self.data_normal = np.load(data_path)  # 输出尺寸(num, 125, 4)

        self.indices = range(len(self))

    def __getitem__(self, index):
        return self.data_normal[index]

    def __len__(self):
        return len(self.data_normal)
