import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import UltraTransformerExpert


class EnsemblePredictor(nn.Module):
    """
    Ensemble model that aggregates predictions from multiple UltraTransformerExpert models.
    It uses confidence-weighted averaging to compute the final prediction.

    Args:
        input_dim (int): Number of input features for each model.
        ensemble_size (int): Number of expert models in the ensemble.
    """
    def __init__(self, input_dim, ensemble_size):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.experts = nn.ModuleList([
            UltraTransformerExpert(input_dim) for _ in range(ensemble_size)
        ])

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the ensemble.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            final (torch.Tensor): Weighted average prediction, shape (batch, 1)
            avg_conf (torch.Tensor): Mean confidence across experts, shape (batch, 1)
            P (torch.Tensor): Raw predictions from each expert, shape (batch, ensemble_size, 1)
            C (torch.Tensor): Confidence scores from each expert, shape (batch, ensemble_size, 1)
        """
        preds, confs = [], []

        for expert in self.experts:
            p, c = expert(x)  # Each returns shape (batch, 1)
            preds.append(p)
            confs.append(c)

        # Stack predictions and confidences: (batch, ensemble_size, 1)
        P = torch.stack(preds, dim=1)
        C = torch.stack(confs, dim=1)

        # Softmax over confidence scores to get weights
        w = F.softmax(C, dim=1)  # (batch, ensemble_size, 1)

        # Weighted sum of predictions
        final = torch.sum(P * w, dim=1)  # (batch, 1)
        avg_conf = torch.mean(C, dim=1)  # (batch, 1)

        return final, avg_conf, P, C

    def get_confidence(self, P: torch.Tensor):
        """
        Estimate ensemble confidence using prediction variance.

        Args:
            P (torch.Tensor): Raw predictions from experts, shape (batch, ensemble_size, 1)

        Returns:
            confidence (float): Average confidence score between 0 (low) and 1 (high)
        """
        # Standard deviation of predictions across experts
        std = torch.std(P, dim=1)  # shape: (batch, 1)
        conf = 1.0 - torch.tanh(std)  # confidence: inverse of uncertainty
        return conf.mean().item()
