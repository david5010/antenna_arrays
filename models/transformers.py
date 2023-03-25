import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerRegressor, self).__init__()
        self.embedding = nn.Linear(input_dim, input_dim)
        self.pos_encoder = PositionalEncoding(input_dim)
        self.transformer = nn.Transformer(
            input_dim, nhead, nlayers, nhid, dropout=dropout
        )
        self.decoder = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        x = self.decoder(x)
        return x.squeeze(-1)

input_dim = 2048
nhead = 8
nhid = 256
nlayers = 3
dropout = 0.1

model = TransformerRegressor(input_dim, nhead, nhid, nlayers, dropout)
