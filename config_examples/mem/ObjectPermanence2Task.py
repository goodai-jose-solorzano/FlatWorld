import math
from typing import List, Tuple

from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement


class ObjectPermanence2Task(FlatWorldConfig):
    def __init__(self):
        super().__init__()

    def get_initial_agent_position(self) -> Tuple[int, int]:
        return 0, 6,

    def get_initial_agent_angle(self) -> float:
        return self.random.uniform(-math.pi / 6, +math.pi / 6)

    def get_elements(self, agent_x: int, agent_y: int) -> List[FlatWorldElement]:
        elements = [
                    FlatWorldElement.fence_brick((6, 4)),
                    FlatWorldElement.fence_brick((7, 4)),
                    FlatWorldElement.final_food((7, 2)),
                    FlatWorldElement.food((5, 6)),
                    ]
        return elements
