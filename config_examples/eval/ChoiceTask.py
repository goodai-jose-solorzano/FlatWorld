import math
from typing import List, Tuple

from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement


class ChoiceTask(FlatWorldConfig):
    def __init__(self):
        super().__init__()

    def get_grid_size(self) -> Tuple[int, int]:
        return 15, 10,

    def get_initial_agent_position(self) -> Tuple[int, int]:
        gw, gh = self.get_grid_size()
        return gw // 2, gh - 1,

    def get_initial_agent_angle(self) -> float:
        return -math.pi / 2

    def get_elements(self, agent_x: int, agent_y: int) -> List[FlatWorldElement]:
        elements = []
        for w in range(3):
            x = w * 4 + 3
            for y in range(2, 5):
                elements.append(FlatWorldElement.wall_brick((x, y)))
        positions = [0, 1, 2, 3]
        element_fns = [FlatWorldElement.final_food, FlatWorldElement.food, FlatWorldElement.final_food_2, FlatWorldElement.fence_brick ]
        self.random.shuffle(positions)
        for p, e_fn in zip(positions, element_fns):
            x = p * 4 + 1
            elements.append(e_fn((x, 3)))
        return elements
