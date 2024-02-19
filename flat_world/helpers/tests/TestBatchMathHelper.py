import math
import unittest

import numpy as np

from flat_world.helpers.batch_math_helper import sq_distances_to_horizontal_lines, min_distances_to_rects


class TestBatchMathHelper(unittest.TestCase):
    def test_sq_distance_to_h_line_1(self):
        x = np.array([1, 2])
        y = np.array([1, 2])
        angles = np.array([math.pi / 2, 0])
        sin_angles = np.sin(angles)
        cos_angles = np.cos(angles)
        slopes = sin_angles / cos_angles
        h_line_y = np.array([-1, 2, 3])
        with np.errstate(divide='ignore', invalid='ignore'):
            sq_d, inter_y = sq_distances_to_horizontal_lines(x, y, slopes, h_line_y)
            expected_sq_d = np.array([[4, np.Inf], [1, 0], [4, np.Inf]], dtype=np.float32)
            expected_inter_y = h_line_y
            assert np.allclose(expected_sq_d, sq_d)
            assert np.allclose(expected_inter_y, h_line_y)

    def test_min_distance_to_rects_1(self):
        x = np.array([1, 2])
        y = np.array([1, 2])
        sin_angles = np.array([math.sin(math.pi / 2), 0])
        cos_angles = np.array([math.cos(math.pi / 2), 1])
        rx1 = np.array([1])
        ry1 = np.array([2])
        rx2 = np.array([2])
        ry2 = np.array([3])
        with np.errstate(divide='ignore', invalid='ignore'):
            distances = min_distances_to_rects(x, y, sin_angles, cos_angles, rx1, ry1, rx2, ry2, inside_ok=True)
            # Shape of distances: 1x2
            expected = np.array([[1, 0]])
            assert np.allclose(expected, distances)

    def test_min_distance_to_rects_2(self):
        # 3 points
        x = np.array([1, 0, 2.0001])
        y = np.array([0.0001, 0.0001, -0.98])
        angles = np.array([math.pi, 0, math.pi / 2])
        sin_angles = np.sin(angles)
        cos_angles = np.cos(angles)
        # 2 rectangles
        rx1 = np.array([-2, +2])
        ry1 = np.array([-1, +0])
        rx2 = np.array([-1, +4])
        ry2 = np.array([+1, +2])
        with np.errstate(divide='ignore', invalid='ignore'):
            distances = min_distances_to_rects(x, y, sin_angles, cos_angles, rx1, ry1, rx2, ry2,
                                               check_direction=False)
            # Shape of distances: 2x3
            expected = np.array([[2, 1, np.Inf], [1, 2, 0.98]])
            assert np.allclose(expected, distances)

    def test_min_distance_to_rects_3(self):
        # Same as above, but checking direction
        # 3 points
        x = np.array([1, 0, 2.0001])
        y = np.array([0.0001, 0.0001, -0.98])
        sin_angles = np.array([math.sin(math.pi), math.sin(0), math.sin(math.pi / 2)])
        cos_angles = np.array([math.cos(math.pi), math.cos(0), math.cos(math.pi / 2)])
        # 2 rectangles
        rx1 = np.array([-2, +2])
        ry1 = np.array([-1, +0])
        rx2 = np.array([-1, +4])
        ry2 = np.array([+1, +2])
        with np.errstate(divide='ignore', invalid='ignore'):
            distances = min_distances_to_rects(x, y, sin_angles, cos_angles, rx1, ry1, rx2, ry2,
                                               check_direction=True)
            # Shape of distances: 2x3
            expected = np.array([[2, np.Inf, np.Inf], [np.Inf, 2, 0.98]])
            assert np.allclose(expected, distances)

    def test_min_distance_to_rects_4(self):
        # Inside not OK
        # 2 points
        x = np.array([0.5, 5])
        y = np.array([0, 0.5])
        angles = np.array([math.pi / 2.0, -math.pi])
        sin_angles = np.sin(angles)
        cos_angles = np.cos(angles)
        # 1 rectangle
        rx1 = np.array([-2])
        ry1 = np.array([-1])
        rx2 = np.array([+1])
        ry2 = np.array([+1])
        with np.errstate(divide='ignore', invalid='ignore'):
            distances = min_distances_to_rects(x, y, sin_angles, cos_angles, rx1, ry1, rx2, ry2,
                                               check_direction=True, inside_ok=False)
            # Shape of distances: 1x2
            expected = np.array([[np.Inf, 4]])
            assert np.allclose(expected, distances)

    def test_min_distance_to_rects_5(self):
        # Same as previous, but inside OK
        # 2 points
        x = np.array([0.5, 5])
        y = np.array([0, 0.5])
        angles = np.array([math.pi / 2.0, -math.pi])
        sin_angles = np.sin(angles)
        cos_angles = np.cos(angles)
        # 1 rectangle
        rx1 = np.array([-2])
        ry1 = np.array([-1])
        rx2 = np.array([+1])
        ry2 = np.array([+1])
        with np.errstate(divide='ignore', invalid='ignore'):
            distances = min_distances_to_rects(x, y, sin_angles, cos_angles, rx1, ry1, rx2, ry2,
                                               check_direction=True, inside_ok=True)
            # Shape of distances: 1x2
            expected = np.array([[1, 4]])
            assert np.allclose(expected, distances)
