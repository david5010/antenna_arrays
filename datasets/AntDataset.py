from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

class AntDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, header = None).values.astype(np.float32)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        features = self.data[index, :-1].reshape(-1,2)
        label = self.data[index, -1]
        return torch.from_numpy(features), torch.from_numpy(np.array(label))
    
class AntDataset2D(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, header = None).values.astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        features = self.data[index, :-1]
        label = self.data[index, -1]
        return torch.from_numpy(features), torch.from_numpy(np.array(label))