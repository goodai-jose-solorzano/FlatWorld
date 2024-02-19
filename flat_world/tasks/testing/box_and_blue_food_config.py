import math
import numpy as np
from typing import Union, List, Tuple
from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement


class BoxAndBlueFoodConfig(FlatWorldConfig):
    # For unit testing

    def get_min_cumulative_reward(self):
        return -3.0

    def get_initial_agent_position(self) -> Tuple[int, int]:
        return 5, 5

    def get_initial_agent_angle(self) -> float:
        return -math.pi / 2

    def get_elements(self, agent_x: int, agent_y: int) -> List[FlatWorldElement]:
        return [
            FlatWorldElement.box((5, 4)),
            FlatWorldElement.final_food_2((5, 3)),
        ]
