import torch
from torch import nn


class WeightedHuberLoss(nn.Module):
    def __init__(self, weights: torch.Tensor, delta=1.0):
        super().__init__()
        self.weight_sum = torch.sum(weights)
        self.weights = weights
        self.delta = delta

    def forward(self, x, target):
        adjusted_weight_sum = self.weight_sum * torch.numel(x) / torch.numel(self.weights)
        diff = target - x
        abs_diff = torch.abs(diff)
        delta = self.delta
        condition = abs_diff < delta
        option1 = 0.5 * diff.pow(2)
        option2 = delta * (abs_diff - 0.5 * delta)
        selection = condition * option1 + ~condition * option2
        return torch.sum(self.weights * selection) / adjusted_weight_sum
