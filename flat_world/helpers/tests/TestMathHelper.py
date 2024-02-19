import math
import unittest

import numpy as np

from flat_world.helpers.math_helper import line_intersection, distance_to_rect_at_angle, norm_angle, \
    norm_angle_differences, line_intersection_with_circle_cao, relationship_with_circle_at_angle


class TestMathHelper(unittest.TestCase):
    def test_norm_angle_1(self):
        a = norm_angle(3 * math.pi / 12 + math.pi * 4, math.pi / 12)
        assert math.isclose(3 * math.pi / 12, a)

    def test_norm_angle_2(self):
        a = norm_angle(-3 * math.pi / 15, math.pi / 15)
        assert math.isclose(-3 * math.pi / 15, a)

    def test_norm_angle_differences(self):
        differences = np.array([-0.5, 0.3 - math.pi * 2, 1.1, -1.4, -1.3 + math.pi * 2])
        norm_differences = norm_angle_differences(differences)
        assert np.allclose(norm_differences, np.array([-0.5, 0.3, 1.1, -1.4, -1.3]))

    def test_line_intersection_1(self):
        line1 = 1, 1, 1, 0
        line2 = 1, 1, 0, 1
        x, y = line_intersection(*line1, *line2)
        assert math.isclose(x, 1)
        assert math.isclose(y, 1)

    def test_line_intersection_2(self):
        line1 = 1, 1, 2, 2
        line2 = -1, -1, 0.57, -2.78
        x, y = line_intersection(*line1, *line2)
        assert math.isclose(x, -1)
        assert math.isclose(y, -1)

    def test_line_intersection_2(self):
        line1 = 0, 3, -3, 1
        line2 = 0, 0, 2, -3
        x, y = line_intersection(*line1, *line2)
        assert math.isclose(-2.571, x, rel_tol=1e-2)
        assert math.isclose(3.857, y, rel_tol=1e-2)

    def test_distance_to_rect_at_angle_1(self):
        rect = 0, 0, 1, 1,
        point = 1, 2,
        angle = math.pi / 2
        d = distance_to_rect_at_angle(*rect, *point, np.sin(angle), np.cos(angle))
        assert math.isclose(d, 1)

    def test_distance_to_rect_at_angle_2(self):
        rect = 1, 2, 2, 3,
        point = 2.5, 2.5,
        angle = math.pi
        d = distance_to_rect_at_angle(*rect, *point, np.sin(angle), np.cos(angle))
        assert math.isclose(d, 0.5)

    def test_distance_to_rect_at_angle_3(self):
        rect = 1, 2, 2, 3,
        point = 2, 1,
        angle = math.pi * 3 / 4
        d = distance_to_rect_at_angle(*rect, *point, np.sin(angle), np.cos(angle))
        assert math.isclose(d, math.sqrt(2))

    def test_distance_to_rect_at_angle_4(self):
        rect = 2, 2, 3, 3,
        point = 1, 2.5,
        angle = math.pi / 8
        d = distance_to_rect_at_angle(*rect, *point, np.sin(angle), np.cos(angle))
        assert math.isclose(d, 1 / math.cos(angle))

    def test_line_intersection_with_circle_cao_1(self):
        radius = 2.5
        line_a = 1.1
        line_b = 0.9
        line_c = 15.0
        p1, p2 = line_intersection_with_circle_cao(radius, line_a, line_b, line_c)
        assert p1 is None
        assert p2 is None

    def test_line_intersection_with_circle_cao_2(self):
        radius = 2.5
        line_a = 0.75
        line_b = -0.75
        line_c = 0
        (x1, y1), (x2, y2) = line_intersection_with_circle_cao(radius, line_a, line_b, line_c)
        v = radius / np.sqrt(2)
        assert math.isclose(v, x1)
        assert math.isclose(v, y1)
        assert math.isclose(v, -x2)
        assert math.isclose(v, -y2)

    def test_line_intersection_with_circle_cao_3(self):
        radius = 2.0
        line_a = 0
        line_b = 1.0
        line_c = -(radius / 2)
        (x1, y1), (x2, y2) = line_intersection_with_circle_cao(radius, line_a, line_b, line_c)
        v = np.cos(math.pi / 6) * radius
        assert math.isclose(v, x1), f'x1={x1}'
        assert math.isclose(-line_c, y1)
        assert math.isclose(v, -x2)
        assert math.isclose(-line_c, y2)

    def test_line_intersection_with_circle_cao_4(self):
        radius = 3.0
        line_a = 1.0
        line_b = 0.0
        line_c = radius / 2
        # eq: X = -1.5
        (x1, y1), (x2, y2) = line_intersection_with_circle_cao(radius, line_a, line_b, line_c)
        v = np.cos(math.pi / 6) * radius
        assert math.isclose(-line_c, x1), f'x1={x1}'
        assert math.isclose(v, y1), f'y1={y1}'
        assert math.isclose(-line_c, x2)
        assert math.isclose(-v, y2)

    def test_line_intersection_with_circle_cao_5(self):
        radius = 1.0
        line_a = 1.0
        line_b = 1.0
        line_c = 0.0
        # eq: X = -1.5
        (x1, y1), (x2, y2) = line_intersection_with_circle_cao(radius, line_a, line_b, line_c)
        v = 1 / np.sqrt(2)
        assert math.isclose(v, x1)
        assert math.isclose(-v, y1), f'y1={y1}'
        assert math.isclose(-v, x2)
        assert math.isclose(v, y2)

    def test_relationship_with_circle_at_angle_1(self):
        cx = -2
        cy = -3
        radius = 1.0
        point_x = 0
        point_y = -3
        sin_angle = math.sin(math.pi)
        cos_angle = math.cos(math.pi)
        dist, ca, ed = relationship_with_circle_at_angle(cx, cy, radius, point_x, point_y, sin_angle, cos_angle)
        assert math.isclose(1, dist), f'dist={dist}'
        assert math.isclose(0, ca, abs_tol=0.00001), f'ca={ca}'
        assert ed, f'ed={ed}'

    def test_relationship_with_circle_at_angle_2(self):
        cx = 3.0
        cy = -2.0
        radius = 0.5
        point_x = 3.0
        point_y = 1.0
        angle = math.pi / 2
        sin_angle = math.sin(angle)
        cos_angle = math.cos(angle)
        dist, ca, ed = relationship_with_circle_at_angle(cx, cy, radius, point_x, point_y, sin_angle, cos_angle)
        assert math.isclose(2.5, dist), f'dist={dist}'
        assert math.isclose(math.pi / 2, ca, abs_tol=0.00001), f'ca={ca}'
        assert not ed, f'ed={ed}'

    def test_relationship_with_circle_at_angle_3(self):
        cx = -2.0
        cy = +2.0
        radius = 1.0
        point_x = 3.0
        point_y = -3.0
        angle = 3 * math.pi / 4
        sin_angle = math.sin(angle)
        cos_angle = math.cos(angle)
        dist, ca, ed = relationship_with_circle_at_angle(cx, cy, radius, point_x, point_y, sin_angle, cos_angle)
        expected_distance = np.sqrt(2 * 5 ** 2) - 1.0
        assert math.isclose(expected_distance, dist), f'dist={dist}, expected={expected_distance}'
        assert math.isclose(-math.pi / 4, ca, abs_tol=0.00001), f'ca={ca}'
        assert ed, f'ed={ed}'

    def test_relationship_with_circle_at_angle_4(self):
        point_x = 2.0
        point_y = 2.0
        ei_x = -2.0
        ei_y = -2.0
        radius = np.sqrt(ei_x ** 2 + ei_y ** 2)
        cx = 0
        cy = -np.sqrt(2 * radius ** 2)
        angle = math.pi / 4
        sin_angle = math.sin(angle)
        cos_angle = math.cos(angle)
        radius_adjusted = radius + 0.0001
        dist, ca, ed = relationship_with_circle_at_angle(cx, cy, radius_adjusted, point_x, point_y, sin_angle, cos_angle)
        expected_distance = radius * 2
        assert math.isclose(expected_distance, dist, abs_tol=0.03), f'dist={dist}, expected={expected_distance}'
        assert math.isclose(3 * math.pi / 4, ca, abs_tol=0.03), f'ca={ca}'
        assert not ed, f'ed={ed}'

    def test_relationship_with_circle_at_angle_5(self):
        point_x = 5.5
        point_y = 9.5
        radius = 0.25
        cx = 7.5
        cy = 1.5
        sin_angle = -0.9713420698132614
        cos_angle = 0.23768589232617332
        dist, ca, ed = relationship_with_circle_at_angle(cx, cy, radius, point_x, point_y, sin_angle, cos_angle)
        assert math.isclose(8.0, dist, abs_tol=0.01)
        assert math.isclose(1.9756, ca, abs_tol=0.01)
        assert ed
