import torch
import torch.nn.functional as F

def robust_loss(pred, target, confidence):
    """Adaptive loss combining MSE & MAE, plus confidence regularization."""
    mse = F.mse_loss(pred, target, reduction="none")
    mae = F.l1_loss(pred, target, reduction="none")
    adapt = confidence * mse + (1 - confidence) * mae
    conf_reg = -torch.log(confidence + 1e-8)
    return adapt.mean() + 0.1 * conf_reg.mean()
