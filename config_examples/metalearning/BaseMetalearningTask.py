import abc
from typing import List, Tuple

import numpy as np
from flat_world.FlatWorldMultiAgentConfig import FlatWorldMultiAgentConfig
from numpy.random import RandomState


class BaseMetalearningTask(FlatWorldMultiAgentConfig, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_reward_upper_bound(self):
        pass

    @staticmethod
    def pos_choice(rs: RandomState, pos_list: List[Tuple[int, int]], size: int):
        pos_idx = rs.choice(range(len(pos_list)), size=size, replace=False)
        selected = np.array(pos_list)[pos_idx]
        return [tuple(pos) for pos in selected]
