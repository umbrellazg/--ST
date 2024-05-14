
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.forward import NaiveFourierKANLayer

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.kan_layer = NaiveFourierKANLayer(d_model, d_ff, gridsize=5)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.kan_layer(x))))