from typing import Tuple, List

import numpy as np

from flat_world.AbstractEnvEntity import AbstractEnvEntity
from flat_world.helpers.PrecalculatedOrientations import PrecalculatedOrientations
from flat_world.helpers.math_helper import norm_angle_differences, distance_to_rect_at_angle

TRANSLUCENT_COLOR = 0x50, 0x50, 0x50, 0x70,
FENCE_COLOR = 0xFF, 0x00, 0x00, 0xFF,
BOX_COLOR = 0x60, 0x50, 0x70, 0xFF,
WALL_COLOR = 0xB5, 0x7F, 0x50, 0xFF,
FINAL_FOOD_COLOR = 0x00, 0xFF, 0x00, 0xFF,
FINAL_FOOD_2_COLOR = 0x50, 0x50, 0xFF, 0xFF,
FOOD_COLOR = 0xFF, 0xFF, 0x00, 0xFF,


class FlatWorldElement(AbstractEnvEntity):
    def __init__(self, color: Tuple[int, int, int, int], is_consumed: bool, initial_position: Tuple[int, int],
                 is_pushable: bool = False, is_displaced: bool = False, is_destroyed: bool = False,
                 reward: float = 0, ends_episode: bool = False, transforms_into_fn=None):
        super().__init__()
        self.color = color
        self.is_consumed = is_consumed
        self.reward = reward
        self.ends_episode = ends_episode
        self.is_pushable = is_pushable
        self.is_displaced = is_displaced
        self.is_destroyed = is_destroyed
        self.position = initial_position
        self.transforms_into_fn = transforms_into_fn

    def move_to(self, position: Tuple[int, int]):
        self.position = position

    @staticmethod
    def at_position(position: Tuple[int, int], **kwargs):
        params = kwargs.copy()
        params['initial_position'] = position
        return FlatWorldElement(**params)

    @staticmethod
    def final_food(position: Tuple[int, int]):
        return FlatWorldElement(color=FINAL_FOOD_COLOR, is_consumed=True, is_displaced=True,
                                reward=1, ends_episode=True,
                                initial_position=position)

    @staticmethod
    def final_food_2(position: Tuple[int, int]):
        return FlatWorldElement(color=FINAL_FOOD_2_COLOR, is_consumed=True, is_displaced=False,
                                reward=0.5, ends_episode=True,
                                initial_position=position,
                                transforms_into_fn=FlatWorldElement.final_food)

    @staticmethod
    def food(position: Tuple[int, int]):
        return FlatWorldElement(color=FOOD_COLOR, is_consumed=True, is_displaced=True,
                                reward=1, ends_episode=False,
                                initial_position=position)

    @staticmethod
    def fence_brick(position: Tuple[int, int]):
        return FlatWorldElement(color=FENCE_COLOR, is_consumed=True, is_destroyed=True,
                                reward=-1, ends_episode=True,
                                initial_position=position)

    @staticmethod
    def translucent_brick(position: Tuple[int, int]):
        return FlatWorldElement(color=TRANSLUCENT_COLOR, is_consumed=False,
                                reward=0, ends_episode=False,
                                initial_position=position)

    @staticmethod
    def box(position: Tuple[int, int]):
        return FlatWorldElement(color=BOX_COLOR, is_consumed=False, is_pushable=True, is_displaced=True,
                                reward=0, ends_episode=False,
                                initial_position=position)

    @staticmethod
    def wall_brick(position: Tuple[int, int]):
        return FlatWorldElement(color=WALL_COLOR, is_consumed=False, is_pushable=False, is_displaced=False,
                                reward=0, ends_episode=False, is_destroyed=True,
                                initial_position=position)
