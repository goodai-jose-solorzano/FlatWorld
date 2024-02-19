import math
from typing import List, Tuple, Union

from flat_world.FlatWorldElement import FlatWorldElement
from flat_world.FlatWorldConfig import FlatWorldConfig


class TwoTranslucentFencesConfig(FlatWorldConfig):
    def get_initial_agent_position(self) -> Tuple[int, int]:
        return 8, 4,

    def get_initial_agent_angle(self) -> float:
        return math.pi

    def get_elements(self, agent_x, agent_y) -> List[object]:
        elements = []
        for b in range(7):
            elements.append(FlatWorldElement.translucent_brick((6, b + 1)))
        for b in range(4):
            elements.append(FlatWorldElement.translucent_brick((3, b + 5)))
        elements.append(FlatWorldElement.final_food((1, 8)))
        elements.append(FlatWorldElement.final_food_2((3, 3)))
        elements.append(FlatWorldElement.food((2, 5)))
        return elements
