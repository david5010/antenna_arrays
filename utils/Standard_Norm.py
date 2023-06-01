import torch
import math
import numpy as np

class ScaleCost:
    """
    Scale Only the cost
    """
    def __init__(self, mean = None, std = None) -> None:
        self.mean = mean
        self.scale = std

    def fit(self, data):
        self.mean = np.mean(data)
        self.scale = np.std(data)
    
    def transform(self, data):
        if self.mean is None or self.scale is None:
            raise ValueError("Scaler has not been fitted")
        return (data - self.mean) / self.scale

    def inverse_transform(self, scaled_data):
        if self.mean is None or self.scale is None:
            raise ValueError("Scaler has not been fitted")
        return (scaled_data * self.scale) + self.mean

    def save_scaler(self, path):
        """
        Saves the fitted parameters
        """
        scaler_attributes = {
            'mean': self.mean,
            'scale': self.scale
        }
        torch.save(scaler_attributes, path)

    def load_scaler(self, path):
        scaler_attributes = torch.load(path)
        self.mean = scaler_attributes['mean']
        self.scale = scaler_attributes['scale']

class StandardNorm:
    """
    Standard Normalizer -> Transforms
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.scale = std

    def fit(self, data):
        num_columns = data.size(1)
        self.mean = torch.zeros(num_columns)
        self.scale = torch.zeros(num_columns)
        
        for column_idx in range(num_columns):
            column = data[:, column_idx]
            non_zero_values = column[column != 0]  # Filter out padding zeros
            
            if non_zero_values.numel() > 0:  # Check if there are non-zero values
                mean = torch.mean(non_zero_values)
                std = torch.std(non_zero_values)
                
                if math.isnan(mean):
                    mean = 0
                if math.isnan(std):
                    std = 1
            else:  # All values in the column are zeros
                mean = 0
                std = 1
                
            self.mean[column_idx] = mean
            self.scale[column_idx] = std
    
    def transform(self, data):
        if self.mean is None or self.scale is None:
            raise ValueError("Scaler has not been fitted")
        return (data - self.mean) / self.scale

    def inverse_transform(self, scaled_data):
        if self.mean is None or self.scale is None:
            raise ValueError("Scaler has not been fitted")
        return (scaled_data * self.scale) + self.mean
    
    def save_scaler(self, path):
        """
        Saves the fitted parameters
        """
        scaler_attributes = {
            'mean': self.mean,
            'scale': self.scale
        }
        torch.save(scaler_attributes, path)

    def load_scaler(self, path):
        scaler_attributes = torch.load(path)
        self.mean = scaler_attributes['mean']
        self.scale = scaler_attributes['scale']
