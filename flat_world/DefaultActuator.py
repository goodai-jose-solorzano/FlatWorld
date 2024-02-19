from typing import Tuple

import numpy as np

from flat_world.AbstractActuator import AbstractActuator
from flat_world.AgentColoring import AgentColoring
from flat_world.BaseEnvCharacter import ACTION_NONE, ACTION_LEFT, ACTION_RIGHT, ACTION_FORWARD, ACTION_BACK, \
    ACTION_TOGGLE
from flat_world.helpers.PrecalculatedOrientations import PrecalculatedOrientations


class DefaultActuator(AbstractActuator):
    def __init__(self, orientation_helper: PrecalculatedOrientations, initial_angle: float, num_orientations: int,
                 movement_step: float):
        super().__init__(orientation_helper)
        self.orientation_index = orientation_helper.get_index_for_angle(initial_angle)
        self.num_orientations = num_orientations
        self.movement_step = movement_step
        self.coloring = AgentColoring.PRIMARY

    def apply_action(self, action: int, position: np.ndarray) -> Tuple[np.ndarray, bool, bool]:
        # Returns the expected new position, and a boolean = movement.
        had_movement = False
        no_action = False
        new_position = position
        if action == ACTION_NONE:
            no_action = True
        elif action == ACTION_LEFT:
            self.orientation_index = round((self.orientation_index - 1) % self.num_orientations)
        elif action == ACTION_RIGHT:
            self.orientation_index = round((self.orientation_index + 1) % self.num_orientations)
        elif action == ACTION_FORWARD or action == ACTION_BACK:
            sign = +1 if action == ACTION_FORWARD else -1
            o_sin, o_cos = self.orientation_helper.get_los_sine_cosine(self.orientation_index)
            step = sign * self.movement_step * np.array([o_cos, o_sin])
            new_position = position + step
            had_movement = True
        elif action == ACTION_TOGGLE:
            self.coloring = self.coloring.opposite()
        else:
            raise Exception(f'Invalid action: {action}')
        return new_position, had_movement, no_action,

    def get_angle(self) -> float:
        return self.orientation_helper.get_angle(self.orientation_index)

    def get_back_color(self) -> Tuple[int, int, int, int]:
        return self.coloring.get_back()

    def get_front_color(self) -> Tuple[int, int, int, int]:
        return self.coloring.get_front()

    def get_cells_sin_cos(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.orientation_helper.get_cell_sines_cosines(self.orientation_index)
