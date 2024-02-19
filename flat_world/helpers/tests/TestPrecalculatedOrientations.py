import math
import unittest

import numpy as np

from flat_world.helpers.PrecalculatedOrientations import PrecalculatedOrientations


class TestPrecalculatedOrientations(unittest.TestCase):
    def test_sin_cos_angles(self):
        obs_resolution = 5
        po = PrecalculatedOrientations(num_orientations=4, obs_resolution=obs_resolution, peripheral_vision_range=math.pi)
        assert po.cell_sines.shape == po.cell_cosines.shape == (4, 5)
        cell_sines, cell_cosines = po.get_cell_sines_cosines(0)
        cell_width = math.pi / obs_resolution
        differences = np.linspace(-math.pi / 2, +math.pi / 2 - cell_width, obs_resolution)
        assert np.allclose(np.sin(differences), cell_sines)
        assert np.allclose(np.cos(differences), cell_cosines)

