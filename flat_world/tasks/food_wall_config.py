import math
from typing import List, Tuple, Union

from flat_world.FlatWorldElement import FlatWorldElement
from flat_world.FlatWorldConfig import FlatWorldConfig


class FoodWallConfig(FlatWorldConfig):
    def get_initial_agent_position(self) -> Tuple[int, int]:
        return 2, 2,

    def get_initial_agent_angle(self) -> float:
        return math.pi / 2

    def get_elements(self, agent_x, agent_y) -> List[object]:
        elements = []
        for b in range(4):
            elements.append(FlatWorldElement.food((6, b + 6)))
        return elements
