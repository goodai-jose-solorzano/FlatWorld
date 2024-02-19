import math
import unittest

import torch
from torch import nn

from flat_world.helpers.WeightedMSELoss import WeightedMSELoss


class TestWeightedMSELoss(unittest.TestCase):
    def test_match(self):
        shape = (4, 5)
        t1 = torch.randn(shape)
        t2 = torch.randn(shape)
        weights = torch.ones_like(t1).bool()
        loss_fn_1 = nn.MSELoss()
        loss_fn_2 = WeightedMSELoss(weights)
        loss_1 = loss_fn_1(t1, t2)
        loss_2 = loss_fn_2(t1, t2)
        assert math.isclose(loss_1.item(), loss_2.item())

    def test_match_2(self):
        shape = (4, 5, 3)
        t1 = torch.randn(shape)
        t2 = torch.randn(shape)
        weights = torch.ones((4, 5)).bool()[:, :, None]
        loss_fn_1 = nn.MSELoss()
        loss_fn_2 = WeightedMSELoss(weights)
        loss_1 = loss_fn_1(t1, t2)
        loss_2 = loss_fn_2(t1, t2)
        assert math.isclose(loss_1.item(), loss_2.item())
