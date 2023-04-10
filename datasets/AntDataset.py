from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import os

class AntDataset(Dataset):
    def __init__(self, csv_file):
        if os.path.splitext(csv_file)[-1] == '.csv':
            self.data = pd.read_csv(csv_file, header = None).values.astype(np.float32)
        else:
            self.data = np.load(csv_file)['data'].astype(np.float32)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        features = self.data[index, :-1]
        label = self.data[index, -1]
        return torch.from_numpy(features), torch.from_numpy(np.array(label))
    
class AntDataset2D(Dataset):
    def __init__(self, csv_file):
        if os.path.splitext(csv_file)[-1] == '.csv':
            self.data = pd.read_csv(csv_file, header= None).values.astype(np.float32)
        else:
            self.data = np.load(csv_file)['data'].astype(np.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the array pattern and target from the CSV data
        array_pattern = self.data[idx, :-1]
        target = self.data[idx, -1]
        
        # Reshape the array pattern and convert to tensor
        sequence_length = 1024
        input_dim = 2
        array_pattern = array_pattern.reshape(input_dim, sequence_length).T
        array_pattern = torch.from_numpy(array_pattern)
        
        # Convert the target to tensor
        target = torch.from_numpy(np.array(target))
        
        return array_pattern, target