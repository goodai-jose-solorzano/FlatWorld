from typing import List, Tuple, Union

from flat_world.FlatWorldElement import FlatWorldElement
from flat_world.FlatWorldConfig import FlatWorldConfig


class DemoConfig(FlatWorldConfig):
    def get_initial_agent_position(self) -> Tuple[int, int]:
        return 4, 6,

    def get_initial_agent_angle(self) -> float:
        return 0

    def get_min_cumulative_reward(self):
        return -3.0

    def get_elements(self, agent_x, agent_y) -> List[object]:
        elements = []
        for b in range(5):
            elements.append(FlatWorldElement.translucent_brick((6, b + 1)))
        for b in range(4):
            elements.append(FlatWorldElement.fence_brick((7, b + 5)))
        elements.append(FlatWorldElement.final_food((8, 3,)))
        elements.append(FlatWorldElement.final_food((6, 7,)))
        elements.append(FlatWorldElement.final_food_2((3, 3,)))
        elements.append(FlatWorldElement.food((9, 7,)))
        elements.append(FlatWorldElement.box((3, 7)))
        elements.append(FlatWorldElement.box((4, 7)))
        return elements
