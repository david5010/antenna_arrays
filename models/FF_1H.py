import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class SimpleNN(nn.Module):
    # in the future, implement one with configuration files for ease
    def __init__(self, input_size, output_size, hidden_size , act_func) -> None:
        super(SimpleNN, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            act_func(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.regressor(x)
    
class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size,1)
    
    def forward(self, x):
        out = self.linear(x)
        return out