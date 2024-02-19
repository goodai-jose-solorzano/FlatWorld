import math
import numpy as np
import torch

SQRT2 = math.sqrt(2)


def weighted_variance(a: np.ndarray, weights: np.ndarray):
    mean = np.average(a, weights=weights)
    diff_sq = (a - mean) ** 2
    return np.sum(diff_sq * weights) / np.sum(weights)


def norm_angle(angle, rotation_step):
    while angle > math.pi:
        angle -= math.pi * 2
    while angle < -math.pi:
        angle += math.pi * 2
    num_splits = round(math.pi * 2 / rotation_step)
    norm_rotation_step = math.pi * 2 / num_splits
    angle_in_steps = round(angle / norm_rotation_step)
    return angle_in_steps * norm_rotation_step


def norm_angle_simple(angle):
    while angle > math.pi:
        angle -= math.pi * 2
    while angle < -math.pi:
        angle += math.pi * 2
    return angle


def norm_angle_differences(differences: np.ndarray):
    over = differences > math.pi
    differences -= math.pi * 2 * over
    under = differences < -math.pi
    differences += math.pi * 2 * under
    return differences


def norm_angle_difference(difference: float):
    while difference > math.pi:
        difference -= math.pi * 2
    while difference < -math.pi:
        difference += math.pi * 2
    return difference


def distance_to_rect_at_angle(x1, y1, x2, y2, x, y, sin_angle, cos_angle):
    # Line parameters
    if abs(sin_angle) < abs(cos_angle):
        line_a = -(sin_angle / cos_angle)
        line_b = 1
        line_c = -(line_a * x + y)
    else:
        line_a = 1
        line_b = -(cos_angle / sin_angle)
        line_c = -(line_b * y + x)

    d_sq_options = []
    # Horizontal line 1 (y = y1)
    inter1x, inter1y = line_intersection_with_horizontal(line_a, line_b, line_c, y1)
    if inter1x and x1 <= inter1x <= x2:
        d = (inter1x - x) ** 2 + ((inter1y - y) ** 2)
        d_sq_options.append(d)
    # Horizontal line 2 (y = y2)
    inter2x, inter2y = line_intersection_with_horizontal(line_a, line_b, line_c, y2)
    if inter2x and y1 <= inter2y <= y2:
        d = (inter2x - x) ** 2 + ((inter2y - y) ** 2)
        d_sq_options.append(d)
    # Vertical line 1  (x = x1)
    inter3x, inter3y = line_intersection_with_vertical(line_a, line_b, line_c, x1)
    if inter3x and x1 <= inter3x <= x2:
        d = (inter3x - x) ** 2 + ((inter3y - y) ** 2)
        d_sq_options.append(d)
    # Vertical line 2  (x = x2)
    inter4x, inter4y = line_intersection_with_vertical(line_a, line_b, line_c, x2)
    if inter4x and y1 <= inter4y <= y2:
        d = (inter4x - x) ** 2 + ((inter4y - y) ** 2)
        d_sq_options.append(d)
    if len(d_sq_options) == 0:
        return None
    min_d_sq = np.min(d_sq_options)
    return math.sqrt(min_d_sq) if min_d_sq >= 0 else None


def line_intersection(line1_start_x, line1_start_y, line1_xf, line1_yf, line2_start_x, line2_start_y, line2_xf, line2_yf):
    # Deprecated
    solution = np.linalg.solve([[line1_xf, -line2_xf], [line1_yf, -line2_yf]],
                               [line2_start_x - line1_start_x, line2_start_y - line1_start_y])
    t1 = solution[0]
    return line1_start_x + t1 * line1_xf, line1_start_y + t1 * line1_yf,


def line_intersection_with_horizontal(line_a: float, line_b: float, line_c: float, other_line_y: float):
    if abs(line_a) < 1e-5:
        return None, None,
    x = - (line_c + line_b * other_line_y) / line_a
    return x, other_line_y,


def line_intersection_with_vertical(line_a: float, line_b: float, line_c: float, other_line_x: float):
    if abs(line_b) < 1e-5:
        return None, None,
    y = - (line_c + line_a * other_line_x) / line_b
    return other_line_x, y,


def line_intersection_with_circle_cao(radius: float, line_a: float, line_b: float, line_c: float):
    '''
    Intersection of line with circle centered at origin.
    '''
    if abs(line_b) < abs(line_a):
        p1, p2 = line_intersection_with_circle_cao(radius, line_b, line_a, line_c)
        if p1 is None:
            return p1, p2,
        (x1, y1), (x2, y2) = p1, p2
        return (y1, x1), (y2, x2),
    # Solves quadratic equation
    line_b_sq = line_b ** 2
    qe_a = line_a ** 2 + line_b_sq
    qe_b = 2 * line_c * line_a
    qe_c = line_c ** 2 - (radius ** 2) * line_b_sq
    discriminant = qe_b ** 2 - 4 * qe_a * qe_c
    if discriminant < 0:
        return None, None,
    qe_root = math.sqrt(discriminant)
    denominator = 2 * qe_a
    # denominator cannot be zero if line_a > 0 or line_b > 0
    x1 = (-qe_b + qe_root) / denominator
    x2 = (-qe_b - qe_root) / denominator
    # line_b cannot be zero because of handling above
    y1 = -(line_c + line_a * x1) / line_b
    y2 = -(line_c + line_a * x2) / line_b
    return (x1, y1), (x2, y2),


def approx_same_direction(sin_angle: float, cos_angle: float, diff_x1: float, diff_y1: float):
    sin_sign = math.copysign(1, sin_angle)
    cos_sign = math.copysign(1, cos_angle)
    if abs(sin_angle) > 0.001:
        if math.copysign(1, diff_y1) != sin_sign:
            return False
    if abs(cos_angle) > 0.001:
        if math.copysign(1, diff_x1) != cos_sign:
            return False
    return True


def relationship_with_circle_at_angle(cx: float, cy: float, radius: float,
                                      point_x: float, point_y: float, sin_angle: float, cos_angle: float):
    '''
    Determines min distance from point to a circle, etc.
    :return: Tuple: Distance, circle's angle of intersection point, boolean -- in expected direction
    '''
    cao_p_x, cao_p_y = point_x - cx, point_y - cy
    if abs(sin_angle) < abs(cos_angle):
        line_a = -(sin_angle / cos_angle)
        line_b = 1
        line_c = -(line_a * cao_p_x + cao_p_y)
    else:
        line_a = 1
        line_b = -(cos_angle / sin_angle)
        line_c = -(line_b * cao_p_y + cao_p_x)
    cao_int_1, cao_int_2 = \
        line_intersection_with_circle_cao(radius, line_a, line_b, line_c)
    if cao_int_1 is None:
        return None, None, None,
    (cao_int_x1, cao_int_y1), (cao_int_x2, cao_int_y2) = cao_int_1, cao_int_2,
    diff_x1, diff_y1 = cao_int_x1 - cao_p_x, cao_int_y1 - cao_p_y
    diff_x2, diff_y2 = cao_int_x2 - cao_p_x, cao_int_y2 - cao_p_y
    dist_sq_1 = diff_x1 ** 2 + diff_y1 ** 2
    dist_sq_2 = diff_x2 ** 2 + diff_y2 ** 2
    expected_direction = approx_same_direction(sin_angle, cos_angle, diff_x1, diff_y1) and \
        approx_same_direction(sin_angle, cos_angle, diff_x2, diff_y2)
    if dist_sq_1 < dist_sq_2:
        ca = math.atan2(cao_int_y1, cao_int_x1)
        return math.sqrt(dist_sq_1), ca, expected_direction,
    else:
        ca = math.atan2(cao_int_y2, cao_int_x2)
        return math.sqrt(dist_sq_2), ca, expected_direction,


