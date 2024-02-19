import math

import numpy as np


class PrecalculatedOrientations:
    def __init__(self, num_orientations: int, obs_resolution: int, peripheral_vision_range: float):
        rotation_step = math.pi * 2 / num_orientations
        los_angles = np.linspace(0, math.pi * 2 - rotation_step, num_orientations)
        cell_width = peripheral_vision_range / obs_resolution
        min_diff = -peripheral_vision_range / 2
        max_diff = -min_diff
        differences = np.linspace(min_diff, max_diff - cell_width, obs_resolution)
        self.differences = differences
        all_angles = los_angles.reshape((num_orientations, 1)) + differences
        # cell_sines and cell_cosines have dimension num_orientations x obs_resolution
        self.cell_sines = np.sin(all_angles)
        self.cell_cosines = np.cos(all_angles)
        # los_sines and los_cosines are arrays of length num_orientations
        self.los_sines = np.sin(los_angles)
        self.los_cosines = np.cos(los_angles)
        self.rotation_step = rotation_step
        self.num_orientations = num_orientations

    def get_cell_sines_cosines(self, orientation_index: int):
        return self.cell_sines[orientation_index], self.cell_cosines[orientation_index],

    def get_cell_sines_cosines_for_angle(self, angle: float):
        target_angles = angle + self.differences
        return np.sin(target_angles), np.cos(target_angles),

    def get_cells_sines_cosines(self, orientation_indexes: np.ndarray):
        # Returns arrays of shape n x obs_resolution,
        # where n is the number of indexes passed (i.e. agents)
        return self.cell_sines[orientation_indexes], self.cell_cosines[orientation_indexes]

    def get_los_sine_cosine(self, orientation_index: int):
        return self.los_sines[orientation_index], self.los_cosines[orientation_index],

    def get_diff_to_angle(self, orientation_index: int, angle: float) -> float:
        o_angle = orientation_index * self.rotation_step
        return angle - o_angle

    def get_diffs_to_angles(self, orientation_index: int, angles: np.ndarray) -> np.ndarray:
        o_angle = orientation_index * self.rotation_step
        return angles - o_angle

    def get_index_for_angle(self, angle: float) -> int:
        return round(angle / self.rotation_step) % self.num_orientations

    def get_angle(self, orientation_index: int) -> float:
        return orientation_index * self.rotation_step
