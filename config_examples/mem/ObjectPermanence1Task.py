import math
from typing import List, Tuple

from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement


class ObjectPermanence1Task(FlatWorldConfig):
    def __init__(self):
        super().__init__()

    def get_initial_agent_position(self) -> Tuple[int, int]:
        return 0, 4,

    def get_initial_agent_angle(self) -> float:
        return self.random.uniform(-math.pi, +math.pi)

    def get_elements(self, agent_x: int, agent_y: int) -> List[FlatWorldElement]:
        elements = [FlatWorldElement.fence_brick((4, 4)),
                    FlatWorldElement.food((4, 3)),
                    FlatWorldElement.food((4, 5)),
                    FlatWorldElement.final_food((9, 4))]
        return elements
