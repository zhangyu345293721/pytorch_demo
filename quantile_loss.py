import torch
import torch.nn as nn
from typing import List

class QuantileLoss(nn.Module):
    def __init__(self, quantile: float):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, pred_y: List[float], y: List[float]) -> torch.Tensor:
        """

        Args:
            pred_y: prediction value
            y: label(ground true)

        Returns:
            quantile loss value
        """
        bias = pred_y - y
        loss = torch.max(bias * (self.quantile - 1), bias * self.quantile)
        return torch.mean(loss)

quantile_loss = QuantileLoss(0.5)
loss = quantile_loss([], [])
print(loss)
