import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import UltraTransformerExpert


class EnsemblePredictor(nn.Module):
    def __init__(self, input_dim, ensemble_size=5):
        super().__init__()
        self.experts = nn.ModuleList([
            UltraTransformerExpert(input_dim) for _ in range(ensemble_size)
        ])

    def forward(self, x):
        preds, confs = [], []
        for m in self.experts:
            p, c = m(x)
            preds.append(p); confs.append(c)
        P = torch.stack(preds, dim=1)    # (batch, M, 1)
        C = torch.stack(confs, dim=1)    # (batch, M, 1)
        w = F.softmax(C, dim=1)
        final = torch.sum(P * w, dim=1)
        avg_conf = torch.mean(C, dim=1)
        return final, avg_conf, P, C
