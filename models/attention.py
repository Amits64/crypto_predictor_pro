import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, seq_len, _ = x.size()
        Q = self.w_q(x).view(b, seq_len, self.n_heads, self.d_k).transpose(1,2)
        K = self.w_k(x).view(b, seq_len, self.n_heads, self.d_k).transpose(1,2)
        V = self.w_v(x).view(b, seq_len, self.n_heads, self.d_k).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / np.sqrt(self.d_k)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)
        out = out.transpose(1,2).contiguous().view(b, seq_len, -1)
        return self.w_o(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
