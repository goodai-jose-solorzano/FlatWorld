import math
from typing import Union, List, Tuple

import numpy as np
import torch

from flat_world.AbstractActuator import AbstractActuator
from flat_world.AgentColoring import AgentColoring
from flat_world.helpers.PrecalculatedOrientations import PrecalculatedOrientations
from flat_world.helpers.math_helper import norm_angle_simple

_single_action_type = Union[List[float], np.ndarray, torch.Tensor]


class BoosterActuator(AbstractActuator):
    def __init__(self, orientation_helper: PrecalculatedOrientations, initial_angle: float, num_boosters: int):
        super().__init__(orientation_helper)
        self.angle = initial_angle
        self.num_boosters = num_boosters
        self.velocity = np.zeros((2,))
        self.angular_velocity = 0
        self.minimal_velocity = 0.03
        self.coloring = AgentColoring.PRIMARY
        clockwise = np.zeros((num_boosters,))
        rot_params = np.zeros((num_boosters, 2, 2,))
        assert num_boosters == 2 or num_boosters == 4, \
            f'The number of boosters requested ({num_boosters}) is not supported!'
        # Boosters:
        # 0: Left
        # 1: Right
        # 2: Bottom
        # 3: Top
        # Power:
        # Positive means fires up for Left and Right boosters
        # Positive means fires right for Bottom and Top boosters
        clockwise[0] = -1
        clockwise[1] = +1
        # Movement in negative direction
        rot_params[0] = np.array([[-1, 0], [0, -1]])
        rot_params[1] = np.array([[-1, 0], [0, -1]])
        if num_boosters == 4:
            clockwise[2] = 1
            clockwise[3] = -1
            # Rotation of -90 degrees
            rot_params[2] = np.array([[0, +1], [-1, 0]])
            rot_params[3] = np.array([[0, +1], [-1, 0]])
        self.clockwise = clockwise
        self.rot_params = rot_params

    def apply_action(self, action: _single_action_type, position: np.ndarray) -> Tuple[np.ndarray, bool, bool]:
        assert hasattr(action, '__len__') and len(action) == self.num_boosters, \
            f'Expected booster action of length {self.num_boosters}'
        sin_angle, cos_angle = math.sin(self.angle), math.cos(self.angle)
        direction = np.array([cos_angle, sin_angle])
        vx, vy = tuple(self.velocity)
        has_movement = vx != 0 or vy != 0
        new_position = position + self.velocity
        self.angle = norm_angle_simple(self.angle + self.angular_velocity)
        self.update_velocity(action, direction)
        self.update_angular_velocity(action)
        no_action = np.all(np.array(action) == 0)
        return new_position, has_movement, no_action,

    def update_angular_velocity(self, action: _single_action_type, friction_factor=0.84,
                                minimal_angular_velocity=0.1 * math.pi / 360, increment_factor=0.1):
        increment = 0
        for i in range(self.num_boosters):
            power = action[i]
            assert -1 <= power <= +1, f'Booster power must be in [-1, +1]. Encountered {power}.'
            increment += power * self.clockwise[i] * increment_factor
        new_av = self.angular_velocity + increment
        new_av *= friction_factor
        if abs(new_av) <= minimal_angular_velocity:
            new_av = 0
        self.angular_velocity = new_av

    def update_velocity(self, action: _single_action_type, direction: np.ndarray,
                        friction_factor=0.84, minimal_velocity=0.001, increment_factor=0.05):
        new_velocity = self.velocity
        direction_matrix = direction[:, None]
        for i in range(self.num_boosters):
            power = action[i]
            assert -1 <= power <= +1, f'Booster power must be in [-1, +1]. Encountered {power}.'
            rot_direction = np.matmul(self.rot_params[i], direction_matrix)
            rot_direction = rot_direction.squeeze(1)
            new_velocity += power * rot_direction * increment_factor
        new_velocity *= friction_factor
        if np.linalg.norm(new_velocity) <= minimal_velocity:
            new_velocity = np.zeros((2,))
        self.velocity = new_velocity

    def get_angle(self) -> float:
        return self.angle

    def get_back_color(self) -> Tuple[int, int, int, int]:
        return self.coloring.get_back()

    def get_front_color(self) -> Tuple[int, int, int, int]:
        return self.coloring.get_front()

    def get_cells_sin_cos(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.orientation_helper.get_cell_sines_cosines_for_angle(self.angle)
