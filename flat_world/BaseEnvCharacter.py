import abc
import math
import numpy as np
from typing import List, Tuple

from numpy.random import RandomState

from flat_world.AbstractEnvEntity import AbstractEnvEntity
from flat_world.AbstractEnvState import AbstractEnvState
from flat_world.helpers.PrecalculatedOrientations import PrecalculatedOrientations
from flat_world.helpers.math_helper import norm_angle, norm_angle_differences, relationship_with_circle_at_angle, \
    norm_angle_simple

DEG_90 = math.pi / 2

ACTION_NONE = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_FORWARD = 3
ACTION_BACK = 4
ACTION_TOGGLE = 5


class BaseEnvCharacter(AbstractEnvEntity):
    def __init__(self, env_state: AbstractEnvState, random: RandomState, index: int,
                 orientation_helper: PrecalculatedOrientations,
                 initial_grid_position: Tuple[int, int],
                 diameter: float, movement_step: float):
        super().__init__()
        self.random = random
        self.env_state = env_state
        self.random = random
        self.index = index
        self.orientation_helper = orientation_helper
        x, y = initial_grid_position
        self.position = np.array([x + 0.5, y + 0.5], dtype=float)
        self.diameter = diameter
        self.movement_step = movement_step
        self.is_done = False

    @abc.abstractmethod
    def has_tail(self) -> bool:
        pass

    @abc.abstractmethod
    def eye_radius(self) -> float:
        pass

    @abc.abstractmethod
    def eye_color(self) -> str:
        pass

    @abc.abstractmethod
    def get_front_color(self) -> Tuple[int, int, int, int]:
        pass

    @abc.abstractmethod
    def get_back_color(self) -> Tuple[int, int, int, int]:
        pass

    @abc.abstractmethod
    def get_angle(self):
        pass
