import abc
from typing import Iterable, Tuple, List

import numpy as np

from flat_world.AbstractEnvEntity import AbstractEnvEntity
from flat_world.FlatWorldElement import FlatWorldElement


class AbstractEnvState(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def remove_elements(self, elements: Iterable[FlatWorldElement]):
        pass

    @abc.abstractmethod
    def get_default_reward(self) -> float:
        pass

    @abc.abstractmethod
    def get_reward_no_action(self) -> float:
        pass

    @abc.abstractmethod
    def get_reward_toggle(self) -> float:
        pass

    @abc.abstractmethod
    def get_grid_size(self) -> Tuple[int, int]:
        pass

    @abc.abstractmethod
    def get_touched_elements(self, rect_x1, rect_y1, rect_x2, rect_y2) -> List[FlatWorldElement]:
        pass

    @abc.abstractmethod
    def attempt_push(self, element: FlatWorldElement, prev_pos: np.ndarray, ax: float, ay: float):
        pass

    @abc.abstractmethod
    def get_elements(self):
        pass

    @abc.abstractmethod
    def get_other_entities(self, agent_index: int) -> List[AbstractEnvEntity]:
        pass

    @abc.abstractmethod
    def get_other_agents(self, agent_index: int) -> List[AbstractEnvEntity]:
        pass

    @abc.abstractmethod
    def get_min_cumulative_reward(self):
        pass

    @abc.abstractmethod
    def get_closest_agent(self, position: np.ndarray) -> Tuple[AbstractEnvEntity, float, float]:
        pass

    @abc.abstractmethod
    def collision_with_element_or_another_predator(self, index: int, position: np.ndarray, diameter: float) -> bool:
        pass

    @abc.abstractmethod
    def get_colliding_predator(self, position: np.ndarray, diameter: float,
                               not_pred_index: int = -1) -> AbstractEnvEntity:
        pass

    @abc.abstractmethod
    def get_predator_reward(self):
        pass
