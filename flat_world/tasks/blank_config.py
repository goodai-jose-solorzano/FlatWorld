from typing import List, Tuple, Union

from flat_world.FlatWorldElement import FlatWorldElement
from flat_world.FlatWorldConfig import FlatWorldConfig


class BlankConfig(FlatWorldConfig):
    def get_min_cumulative_reward(self):
        return -10.0

    def get_initial_agent_position(self) -> Tuple[int, int]:
        return 4, 6,

    def get_initial_agent_angle(self) -> float:
        return 0

    def get_elements(self, agent_x, agent_y) -> List[object]:
        return []
