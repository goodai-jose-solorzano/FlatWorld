import math
from typing import Union, List, Tuple

import numpy as np
import torch

from flat_world.AbstractActuator import AbstractActuator
from flat_world.AgentColoring import AgentColoring
from flat_world.helpers.PrecalculatedOrientations import PrecalculatedOrientations
from flat_world.helpers.math_helper import norm_angle_simple

_single_action_type = Union[List[float], np.ndarray, torch.Tensor]


class DualActuator(AbstractActuator):
    def __init__(self, orientation_helper: PrecalculatedOrientations, initial_angle: float, angular_step=math.pi / 30):
        super().__init__(orientation_helper)
        self.angle = round(initial_angle / angular_step) * angular_step
        self.speed = 0
        self.minimal_velocity = 0.01
        self.coloring = AgentColoring.PRIMARY
        self.angular_step = angular_step

    def apply_action(self, action: _single_action_type, position: np.ndarray) -> Tuple[np.ndarray, bool, bool]:
        motion_action, rot_action = action[0], action[1],
        if rot_action != 0:
            change = self.angular_step if rot_action == 1 else -self.angular_step
            self.angle = norm_angle_simple(self.angle + change)
        sin_angle, cos_angle = math.sin(self.angle), math.cos(self.angle)
        direction = np.array([cos_angle, sin_angle])
        self.update_speed(action)
        s = self.speed
        has_movement = s != 0
        new_position = position + s * direction
        no_action = motion_action == 0 and rot_action == 0
        return new_position, has_movement, no_action,

    def update_speed(self, action: _single_action_type,
                     friction_factor=0.77, minimal_velocity=0.03, increment_factor=0.077):
        new_speed = self.speed
        motion_action = action[0]
        power = 0 if motion_action == 0 else 1 if motion_action == 1 else -1
        new_speed += power * increment_factor
        new_speed *= friction_factor
        if new_speed <= minimal_velocity:
            new_speed = 0
        self.speed = new_speed

    def get_angle(self) -> float:
        return self.angle

    def get_back_color(self) -> Tuple[int, int, int, int]:
        return self.coloring.get_back()

    def get_front_color(self) -> Tuple[int, int, int, int]:
        return self.coloring.get_front()

    def get_cells_sin_cos(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.orientation_helper.get_cell_sines_cosines_for_angle(self.angle)
