import torch
import torch.nn as nn

class TransformerRegressor(nn.Module):
    def __init__(self, d_model,
                nhead, num_encoder_layers,
                num_decoder_layers,
                dim_feedforward,
                dropout,
                input_dim=2,
                batch_first=True) -> None:
        super().__init__()

        self.name = 'Transformer'

        # Linear layer to project input to d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.target_proj = nn.Linear(1, d_model)
        # base of the transformer
        self.transformer = nn.Transformer(d_model, nhead,
                                          num_encoder_layers,
                                          num_decoder_layers,
                                          dim_feedforward,
                                          dropout,
                                          batch_first=batch_first)
        # output layer to 1
        self.fc = nn.Linear(d_model, 1)

    def forward(self, src, tgt):
        # Project input to d_model
        # src = self.input_proj(src)
        # # Generate masks
        tgt = tgt.reshape(-1,1)
        # src = self.input_proj(src)
        tgt = self.target_proj(tgt)
        src_padding_mask = (src[..., 0] == 0) & (src[..., 1] == 0)
        # If i do self.transformer(src,src, src_key_padding_mask = src_padding_mask) works but not good results?
        out = self.transformer(src,tgt, src_key_padding_mask = src_padding_mask)
        out = self.fc(out[:,-1,:])
        return out

        
        # src_mask = src_padding_mask.unsqueeze(1).unsqueeze(2)
        # src_mask = src_mask.float().masked_fill(src_mask == 1, float('-inf')).masked_fill(src_mask == 0, float(0.0))

        # # Repeat target tensor to match src tensor shape
        # tgt_repeated = tgt.unsqueeze(1).repeat(1, src.size(1), 1)

        # # Pass through transformer
        # out = self.transformer(src, tgt_repeated, src_mask=src_mask, src_key_padding_mask=src_padding_mask)

        # # Apply linear layer to the last output of the transformer
        # out = self.fc(out[:, -1, :])
        # return out



# import torch
# import torch.nn as nn
# import math

# class TransformerRegressor(nn.Module):

#     def __init__(self, d_model,
#                 nhead, num_encoder_layers,
#                 num_decoder_layers,
#                 dim_feedforward,
#                 dropout,
#                 batch_first = True) -> None:
#         super().__init__()
        
#         self.name = 'Transformer'

#         # base of the transformer
#         self.transformer = nn.Transformer(d_model, nhead,
#                                           num_encoder_layers,
#                                           num_decoder_layers,
#                                           dim_feedforward,
#                                           dropout,
#                                           batch_first=batch_first)
#         # output layer to 1
#         self.fc = nn.Linear(d_model, 1)

#     def forward(self, src, tgt):
#         # Generate masks
#         src_padding_mask = (src[..., 0] == 0) & (src[..., 1] == 0)
#         src_mask = src_padding_mask.unsqueeze(1).unsqueeze(2)
#         src_mask = src_mask.float().masked_fill(src_mask == 1, float('-inf')).masked_fill(src_mask == 0, float(0.0))
        
#         # Permutation invariant: sum the source sequence along the time axis
#         src = src.sum(dim=1)
        
#         # Repeat target tensor to match src tensor shape
#         tgt_repeated = tgt.unsqueeze(1).repeat(1, src.size(1), 1)
        
#         # Pass through transformer
#         out = self.transformer(src, tgt_repeated, src_mask=src_mask, src_key_padding_mask=src_padding_mask)
        
#         # Apply linear layer to the last output of the transformer
#         out = self.fc(out[:, -1, :])
#         return out
