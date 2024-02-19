import abc
from typing import Union, List, Tuple

import numpy as np
import torch

from flat_world.helpers.PrecalculatedOrientations import PrecalculatedOrientations

_single_action_type = Union[int, List[int], List[float], np.ndarray, torch.Tensor]


class AbstractActuator(abc.ABC):
    def __init__(self, orientation_helper: PrecalculatedOrientations):
        self.orientation_helper = orientation_helper

    @abc.abstractmethod
    def apply_action(self, action: _single_action_type, position: np.ndarray) -> Tuple[np.ndarray, bool, bool]:
        '''
        Applies an action.
        Returns the expected new position, a boolean = movement, and a boolean = no action.
        '''
        pass

    @abc.abstractmethod
    def get_angle(self) -> float:
        pass

    @abc.abstractmethod
    def get_back_color(self) -> Tuple[int, int, int, int]:
        pass

    @abc.abstractmethod
    def get_front_color(self) -> Tuple[int, int, int, int]:
        pass

    @abc.abstractmethod
    def get_cells_sin_cos(self) -> Tuple[np.ndarray, np.ndarray]:
        pass
