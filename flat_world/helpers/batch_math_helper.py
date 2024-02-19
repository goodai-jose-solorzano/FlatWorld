import numpy as np


def mult_zero_preferred(x: np.ndarray, y: np.ndarray):
    is_zero = x == 0
    result = x * y
    result[is_zero] = 0
    return result


def sq_distances_to_horizontal_lines(x: np.ndarray, y: np.ndarray, slopes: np.ndarray,
                                     other_line_y: np.ndarray):
    # Given:
    #   n = Number of points (i.e. agents)
    #   m = Number of horizontal lines (i.e. elements)
    # Parameter dimensions are:
    #   x: n
    #   y: n
    #   slopes: n
    #   other_line_y: m
    # Return value dimensions: m x n, n x m

    y_diffs = np.subtract.outer(other_line_y, y)
    inter_x = x + mult_zero_preferred(y_diffs, 1.0 / slopes)
    return (x - inter_x) ** 2 + y_diffs ** 2, inter_x.transpose(),


def sq_distances_to_vertical_lines(x: np.ndarray, y: np.ndarray, slopes: np.ndarray,
                                   other_line_x: np.ndarray):
    x_diffs = np.subtract.outer(other_line_x, x)
    inter_y = y + mult_zero_preferred(x_diffs, slopes)
    return (y - inter_y) ** 2 + x_diffs ** 2, inter_y.transpose(),


def min_distances_to_rects(x: np.ndarray, y: np.ndarray, sin_angle: np.ndarray, cos_angle: np.ndarray,
                           rx1: np.ndarray, ry1: np.ndarray, rx2: np.ndarray, ry2: np.ndarray,
                           check_direction=True, inside_ok=False):
    # Requires: rx1 < rx2, ry1 < ry2
    # For n agents and m elements, this returns an array with shape (m, n) with distances.
    # Distances are np.Inf if there's no intersection or if the element is in the wrong direction.
    slopes = sin_angle / cos_angle
    sdh1, int_x_h1 = sq_distances_to_horizontal_lines(x, y, slopes, ry1)
    sdh2, int_x_h2 = sq_distances_to_horizontal_lines(x, y, slopes, ry2)
    sdv1, int_y_v1 = sq_distances_to_vertical_lines(x, y, slopes, rx1)
    sdv2, int_y_v2 = sq_distances_to_vertical_lines(x, y, slopes, rx2)
    invalid_h1 = ((rx1 > int_x_h1) | (int_x_h1 > rx2)).transpose()
    invalid_h2 = ((rx1 > int_x_h2) | (int_x_h2 > rx2)).transpose()
    invalid_v1 = ((ry1 > int_y_v1) | (int_y_v1 > ry2)).transpose()
    invalid_v2 = ((ry1 > int_y_v2) | (int_y_v2 > ry2)).transpose()
    if check_direction:
        invalid_dir_h1 = np.sign(np.subtract.outer(ry1, y)) != np.sign(sin_angle)
        invalid_dir_h2 = np.sign(np.subtract.outer(ry2, y)) != np.sign(sin_angle)
        invalid_dir_v1 = np.sign(np.subtract.outer(rx1, x)) != np.sign(cos_angle)
        invalid_dir_v2 = np.sign(np.subtract.outer(rx2, x)) != np.sign(cos_angle)
        invalid_h1 |= invalid_dir_h1
        invalid_h2 |= invalid_dir_h2
        invalid_v1 |= invalid_dir_v1
        invalid_v2 |= invalid_dir_v2
        if not inside_ok:
            invalid_h1 |= invalid_dir_h2
            invalid_h2 |= invalid_dir_h1
            invalid_v1 |= invalid_dir_v2
            invalid_v2 |= invalid_dir_v1
    sdh1[invalid_h1] = np.Inf
    sdh2[invalid_h2] = np.Inf
    sdv1[invalid_v1] = np.Inf
    sdv2[invalid_v2] = np.Inf
    h_min = np.minimum(sdh1, sdh2)
    v_min = np.minimum(sdv1, sdv2)
    min_sq_d = np.minimum(h_min, v_min)
    return np.sqrt(min_sq_d)
