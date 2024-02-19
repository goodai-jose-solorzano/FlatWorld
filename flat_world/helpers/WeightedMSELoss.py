import torch
from torch import nn


class WeightedMSELoss(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.weight_sum = torch.sum(weights)
        self.weights = weights

    def forward(self, x, target):
        adjusted_weight_sum = self.weight_sum * torch.numel(x) / torch.numel(self.weights)
        return torch.sum(self.weights * ((target - x) ** 2)) / adjusted_weight_sum
