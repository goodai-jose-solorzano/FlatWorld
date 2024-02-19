import math
import numpy as np
from typing import Tuple, List, Any, Union

import torch
from numpy.random import RandomState

from flat_world.AbstractActuator import AbstractActuator
from flat_world.AbstractEnvEntity import AbstractEnvEntity
from flat_world.AbstractEnvState import AbstractEnvState
from flat_world.ActuatorType import ActuatorType
from flat_world.AgentColoring import AgentColoring
from flat_world.BaseEnvCharacter import BaseEnvCharacter, ACTION_NONE, ACTION_LEFT, ACTION_RIGHT, ACTION_FORWARD, \
    ACTION_BACK, ACTION_TOGGLE
from flat_world.BoosterActuator import BoosterActuator
from flat_world.DefaultActuator import DefaultActuator
from flat_world.DualActuator import DualActuator
from flat_world.PushException import PushException
from flat_world.helpers.PrecalculatedOrientations import PrecalculatedOrientations
from flat_world.helpers.math_helper import norm_angle, SQRT2

_single_action_type = Union[int, List[int], List[float], np.ndarray, torch.Tensor]


class AgentState(BaseEnvCharacter):
    def __init__(self, env_state: AbstractEnvState, random: RandomState, index: int,
                 actuator_type: ActuatorType,
                 orientation_helper: PrecalculatedOrientations,
                 peripheral_vision_range: float, num_orientations: int,
                 initial_grid_position: Tuple[int, int], initial_angle: float,
                 movement_step: float, agent_diameter: float, obs_resolution: int):
        super().__init__(env_state, random, index, orientation_helper,
                         initial_grid_position, agent_diameter, movement_step)
        self.peripheral_vision_range = peripheral_vision_range
        self.diameter = agent_diameter
        self.obs_resolution = obs_resolution
        self.cumulative_reward = 0
        self.done_type = None
        if actuator_type == ActuatorType.DEFAULT:
            self.actuator: AbstractActuator = DefaultActuator(orientation_helper, initial_angle, num_orientations,
                                                              movement_step)
        elif actuator_type == ActuatorType.DUAL:
            self.actuator: AbstractActuator = DualActuator(orientation_helper, initial_angle)
        else:
            num_boosters = actuator_type.get_num_boosters()
            assert num_boosters != 0
            self.actuator = AgentState._booster_actuator(num_boosters, orientation_helper, initial_angle)

    @staticmethod
    def _booster_actuator(num_boosters: int, orientation_helper: PrecalculatedOrientations,
                          initial_angle: float) -> AbstractActuator:
        return BoosterActuator(orientation_helper, initial_angle, num_boosters)

    def has_tail(self) -> bool:
        return True

    def eye_radius(self) -> float:
        return 1.0

    def eye_color(self) -> str:
        return 'black'

    def get_back_color(self):
        return self.actuator.get_back_color()

    def get_front_color(self):
        return self.actuator.get_front_color()

    def get_cells_sin_cos(self):
        return self.actuator.get_cells_sin_cos()

    def get_info(self):
        dt = self.done_type
        return {'completion': dt} if dt else {}

    def apply_action(self, action: _single_action_type):
        if self.is_done:
            return 0
        default_reward = self.env_state.get_default_reward()
        new_position, had_movement, no_action = self.actuator.apply_action(action, self.position)
        applied_reward = self.env_state.get_reward_no_action() if no_action else default_reward
        if had_movement:
            reward_done = self.apply_new_agent_position(new_position, applied_reward)
        else:
            reward_done = applied_reward, False,
        reward, done = reward_done
        self.cumulative_reward += reward
        if not done:
            cp: Any = self.env_state.get_colliding_predator(self.position, self.diameter)
            if cp:
                done = True
                self.done_type = 'killed'
                reward += self.env_state.get_predator_reward()
                cp.reset_after_eat_count()
            if not done and self.cumulative_reward < self.env_state.get_min_cumulative_reward():
                done = True
                self.done_type = 'ran_out'
        else:
            self.done_type = 'killed' if reward < 0 else 'success'
        self.is_done = done
        return reward

    def collides_with_another(self, new_position: np.ndarray):
        other_agents = self.env_state.get_other_agents(self.index)
        min_d_sq = self.diameter ** 2
        oa_s: AgentState
        for oa_s in other_agents:
            d_sq = np.sum((oa_s.position - new_position) ** 2)
            if d_sq < min_d_sq:
                return True
        return False

    def apply_new_agent_position(self, new_position: np.ndarray, default_reward: float):
        x, y = tuple(new_position)
        g_w, g_h = self.env_state.get_grid_size()
        d = self.diameter
        r = d / (2 * SQRT2)
        x1, y1, x2, y2 = x - r, y - r, x + r, y + r
        if x1 < 0 or y1 < 0:
            return default_reward, False,
        if x2 > g_w or y2 > g_h:
            return default_reward, False,
        if self.collides_with_another(new_position):
            return default_reward, False,
        touched_elements = self.env_state.get_touched_elements(x1, y1, x2, y2)
        cannot_move = False
        reward = 0
        end_of_episode = False
        consumed_elements = []
        pushed_elements = []
        for te in touched_elements:
            if te.is_consumed:
                consumed_elements.append(te)
                reward += te.reward
                if te.ends_episode:
                    end_of_episode = True
            elif te.is_pushable:
                pushed_elements.append(te)
            else:
                cannot_move = True
        if cannot_move:
            return default_reward, False,
        for pe in pushed_elements:
            try:
                self.env_state.attempt_push(pe, self.position, x, y)
            except PushException:
                return default_reward, False,
        self.env_state.remove_elements(consumed_elements)
        self.position = new_position
        if reward == 0:
            reward = default_reward
        return reward, end_of_episode,

    def get_angle(self):
        return self.actuator.get_angle()

