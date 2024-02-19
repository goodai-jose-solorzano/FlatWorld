import math
from typing import Tuple

import numpy as np
from numpy.random import RandomState

from flat_world.AbstractEnvState import AbstractEnvState
from flat_world.BaseEnvCharacter import BaseEnvCharacter, ACTION_LEFT, ACTION_RIGHT, ACTION_FORWARD, ACTION_NONE
from flat_world.PredatorLevel import PredatorLevel
from flat_world.helpers.PrecalculatedOrientations import PrecalculatedOrientations
from flat_world.helpers.math_helper import norm_angle_difference

PREDATOR_ACTIONS = [ACTION_LEFT, ACTION_RIGHT, ACTION_FORWARD]


class PredatorState(BaseEnvCharacter):
    def __init__(self, env_state: AbstractEnvState, random: RandomState, index: int,
                 orientation_helper: PrecalculatedOrientations,
                 num_orientations: int,
                 initial_grid_position: Tuple[int, int], initial_orientation_index: int,
                 diameter: float, agent_movement_step: float, predator_level: PredatorLevel):
        movement_step = agent_movement_step * predator_level.get_speed_factor()
        super().__init__(env_state, random, index, orientation_helper,
                         initial_grid_position, diameter, movement_step)
        self.orientation_index = initial_orientation_index
        self.num_orientations = num_orientations
        self.prev_action = self.random.choice(PREDATOR_ACTIONS)
        self.prob_follow_agents = predator_level.get_prob_follow_agents()
        self.prob_continue_forward_action = predator_level.get_prob_continue_forward_action()
        self.prob_continue_rotation_action = predator_level.get_prob_continue_rotation_action()
        self.inactive_steps_after_eating = predator_level.get_inactive_steps_after_eating()
        self.after_eat_count = self.inactive_steps_after_eating

    def has_tail(self) -> bool:
        return False

    def eye_radius(self) -> float:
        return 2.0

    def eye_color(self) -> str:
        return 'green'

    def reset_after_eat_count(self):
        self.after_eat_count = 0

    def get_back_color(self) -> Tuple[int, int, int, int]:
        return 0xFF, 0xA0, 0x00, 0xFF,

    def get_front_color(self) -> Tuple[int, int, int, int]:
        return 0xFF, 0xAA, 0x20, 0xFF,

    def get_attack_action(self):
        agent_state, diff_x, diff_y = self.env_state.get_closest_agent(self.position)
        if agent_state is None:
            return self.random.choice(PREDATOR_ACTIONS)
        prey_angle = math.atan2(diff_y, diff_x)
        angle_diff = norm_angle_difference(
            self.orientation_helper.get_diff_to_angle(self.orientation_index, prey_angle))
        if np.abs(angle_diff) < math.pi / 6:
            return ACTION_FORWARD
        elif angle_diff > 0:
            return ACTION_RIGHT
        else:
            return ACTION_LEFT

    def apply_stochastic_action(self):
        if self.is_done:
            return
        self.after_eat_count += 1
        if self.after_eat_count <= self.inactive_steps_after_eating:
            return
        if self.prev_action == ACTION_FORWARD and self.random.uniform() < self.prob_continue_forward_action:
            action = self.prev_action
        elif self.prev_action != ACTION_FORWARD and self.random.uniform() < self.prob_continue_rotation_action:
            action = self.prev_action
        elif self.random.uniform() < self.prob_follow_agents:
            action = self.get_attack_action()
        else:
            action = self.random.choice(PREDATOR_ACTIONS)
        if action == ACTION_NONE:
            pass
        elif action == ACTION_LEFT:
            self.orientation_index = round((self.orientation_index - 1) % self.num_orientations)
        elif action == ACTION_RIGHT:
            self.orientation_index = round((self.orientation_index + 1) % self.num_orientations)
        elif action == ACTION_FORWARD:
            o_sin, o_cos = self.orientation_helper.get_los_sine_cosine(self.orientation_index)
            step = self.movement_step * np.array([o_cos, o_sin])
            new_position = self.position + step
            if self.env_state.collision_with_element_or_another_predator(self.index, new_position, self.diameter):
                return
            self.position = new_position
        else:
            raise Exception(f'Invalid action: {action}')

    def get_angle(self):
        return self.orientation_helper.get_angle(self.orientation_index)
