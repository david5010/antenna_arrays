import torch.nn as nn
import torch

class DeepSetRegression(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, pooling='max', dim_hidden=128):
        super(DeepSetRegression, self).__init__()
        self.name = 'Deepset'
        self.pooling = pooling
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))

    def forward(self, X):
        mask = torch.any(X != 0, axis=-1)
        X = self.enc(X)
        if self.pooling == 'sum':
            X = X.sum(-2)
        elif self.pooling == 'max':
            X,_ = X.max(-2)
        elif self.pooling == 'mean':
            X  = X.mean(-2)
        elif self.pooling == 'min':
            X, _ = X.min(-2)
        elif self.pooling == 'l1':
            X = torch.sum(torch.abs(X), dim = -2)
        elif self.pooling == 'robust_mean':
            X_sum = torch.sum(X, dim=-2)
            mask_sum = torch.sum(mask, dim=-1, keepdim=True)
            mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
            X_mean = X_sum / mask_sum.float()
            X = X_mean
        # pooling can change. Maybe max or L2-norm since we have 0 paddings
        # X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output) # can change the output
        X = self.dec(X)
        return X