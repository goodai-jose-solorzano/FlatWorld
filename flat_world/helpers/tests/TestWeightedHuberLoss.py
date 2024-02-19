import math
import unittest

import torch
from torch import nn

from flat_world.helpers.WeightedHuberLoss import WeightedHuberLoss
from flat_world.helpers.WeightedMSELoss import WeightedMSELoss


class TestWeightedHuberLoss(unittest.TestCase):
    def test_match(self):
        shape = (7, 11)
        t1 = torch.randn(shape)
        t2 = torch.randn(shape)
        weights = torch.ones_like(t1).bool()
        loss_fn_1 = nn.HuberLoss()
        loss_fn_2 = WeightedHuberLoss(weights)
        loss_1 = loss_fn_1(t1, t2)
        loss_2 = loss_fn_2(t1, t2)
        assert math.isclose(loss_1.item(), loss_2.item())
