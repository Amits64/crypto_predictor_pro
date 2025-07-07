import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.attention import PositionalEncoding

class UltraTransformerExpert(nn.Module):
    def __init__(self, input_dim, d_model=256, n_heads=8, n_layers=6, dropout=0.1):
        super().__init__()
        self.project = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, n_layers)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model//2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model//2, 1)
        )
        self.conf_head = nn.Sequential(
            nn.Linear(d_model, d_model//2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.project(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        last = x[:, -1, :]
        return self.pred_head(last), self.conf_head(last)
