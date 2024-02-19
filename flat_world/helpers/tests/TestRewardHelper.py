import unittest

import numpy as np

from flat_world.helpers.reward_helper import get_discounted_reward_matrix, get_future_reward_matrix


class TestRewardHelper(unittest.TestCase):
    def get_drm(self, reward_matrix: np.ndarray, gamma: float):
        m, n = reward_matrix.shape
        drm = np.zeros((m, n,))
        for i in range(m):
            sum = np.zeros((n,))
            factor = 1.0
            for j in range(i, m):
                sum += factor * reward_matrix[j, :]
                factor *= gamma
            drm[i, :] = sum
        return drm

    def test_future_reward_matrix(self):
        rm = np.random.normal(size=(4, 5))
        frm_wi = get_future_reward_matrix(rm, includes_immediate=True)
        frm_ni = get_future_reward_matrix(rm, includes_immediate=False)
        assert np.allclose(rm + frm_ni, frm_wi)

    def test_discounted_reward_1(self):
        # reward matrix shape: (m, n,)
        rm = np.array([
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [0.5, 2.5],
        ])
        gamma = 0.9
        drm = get_discounted_reward_matrix(rm, gamma)
        expected_drm = self.get_drm(rm, gamma)
        assert np.allclose(drm, expected_drm)

    def test_discounted_reward_2(self):
        m = 100
        n = 10
        rm = np.random.normal(size=(m, n,))
        gamma = 0.9
        drm = get_discounted_reward_matrix(rm, gamma)
        expected_drm = self.get_drm(rm, gamma)
        assert np.allclose(drm, expected_drm)

    def test_discounted_reward_3(self):
        m = 2000
        n = 10
        rm = np.random.normal(size=(m, n,))
        gamma = 0.9
        drm = get_discounted_reward_matrix(rm, gamma)
        expected_drm = self.get_drm(rm, gamma)
        assert np.allclose(drm, expected_drm)

    def test_discounted_reward_4(self):
        m = 40
        n = 1
        rm = np.random.normal(size=(m, n,))
        gamma = 0.001
        drm = get_discounted_reward_matrix(rm, gamma)
        expected_drm = self.get_drm(rm, gamma)
        assert np.allclose(drm, expected_drm, atol=0.01)
