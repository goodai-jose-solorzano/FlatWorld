from typing import List, Tuple, Union

import numpy as np

from flat_world.FlatWorldElement import FlatWorldElement
from flat_world.FlatWorldConfig import FlatWorldConfig


class HallwayTask(FlatWorldConfig):
    def __init__(self, level=5):
        super().__init__()
        self.level = level

    def get_initial_agent_position(self) -> Tuple[int, int]:
        return 5, 8,

    def get_initial_agent_angle(self) -> float:
        return 0

    def get_elements(self, agent_x, agent_y) -> List[object]:
        elements = [FlatWorldElement.final_food((9, 8))]
        upper_elements = []
        # Walls:
        for b in range(3):
            elements.append(FlatWorldElement.translucent_brick((7, 9 - b)))
            upper_elements.append(FlatWorldElement.wall_brick((7, b)))
        upper_elements.append(FlatWorldElement.wall_brick((7, 3)))
        # Entrance:
        for b in range(3):
            elements.append(FlatWorldElement.wall_brick((9 - b - 2, 6)))
            upper_elements.append(FlatWorldElement.wall_brick((9 - b - 2, 4)))
        if self.level == 0:
            elements.extend(upper_elements[:2])
        elif self.level == 1:
            elements.extend(upper_elements[:4])
        elif self.level == 2:
            elements.extend(upper_elements[:6])
        else:
            elements.extend(upper_elements)
        return elements

    def get_upper_prob(self):
        probs = [0.1, 0.3, 0.75, 1.0]
        level = max(0, min(len(probs), self.level))
        return probs[level]
