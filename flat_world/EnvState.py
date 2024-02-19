import math
from typing import Tuple, List, Iterable, Union

import numpy as np
import torch
from numpy.random import RandomState

from flat_world.AbstractEnvEntity import AbstractEnvEntity
from flat_world.AbstractEnvState import AbstractEnvState
from flat_world.ActuatorType import ActuatorType
from flat_world.AgentState import AgentState
from flat_world.BaseEnvCharacter import BaseEnvCharacter
from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement
from flat_world.PredatorLevel import PredatorLevel
from flat_world.PredatorState import PredatorState
from flat_world.PushException import PushException
from flat_world.helpers.PrecalculatedOrientations import PrecalculatedOrientations
from flat_world.helpers.batch_math_helper import min_distances_to_rects
from flat_world.helpers.math_helper import SQRT2

_single_action_type = Union[int, List[float], np.ndarray, torch.Tensor]
_multi_action_type = Union[List[_single_action_type], np.ndarray, torch.Tensor]


class EnvState(AbstractEnvState):
    def __init__(self, config: FlatWorldConfig, random: RandomState,
                 orientation_helper: PrecalculatedOrientations,
                 num_agents: int, obs_resolution: int,
                 default_reward: float, reward_no_action: float,
                 reward_toggle: float, reward_predator: float,
                 min_cumulative_reward: float,
                 predator_level: PredatorLevel,
                 actuator_type: ActuatorType,
                 squeeze: bool = False):
        super().__init__()
        self.default_reward = default_reward
        self.reward_no_action = reward_no_action
        self.reward_toggle = reward_toggle
        self.reward_predator = reward_predator
        self.min_cumulative_reward = min_cumulative_reward
        self.obs_resolution = obs_resolution
        self.orientation_helper = orientation_helper
        self.squeeze = squeeze
        self.elements_grid = None
        self.random = random
        self.is_batched = num_agents > 1 or not squeeze
        agent_peripheral_vision_range = config.get_peripheral_vision_range()
        rotation_step = config.get_rotation_step()
        num_orientations = np.round(math.pi * 2 / rotation_step)
        movement_step = config.get_movement_step()
        self.grid_size = config.get_grid_size()
        gw, gh = self.grid_size
        self.agent_diameter = 0.5
        agent_states = []
        agent_pos_list = config.get_initial_agent_positions(num_agents)
        # TODO should pass agent positions to get angles
        agent_iangle_list = config.get_initial_agent_angles(num_agents)
        for i in range(num_agents):
            agent_initial_pos = agent_pos_list[i]
            ax, ay = agent_initial_pos
            if ax < 0 or ay < 0 or ax >= gw or ay >= gh:
                raise Exception(f'Invalid initial agent position: {agent_initial_pos}')
            agent_initial_angle = agent_iangle_list[i]
            agent_states.append(AgentState(self, self.random, i, actuator_type,
                                           orientation_helper, agent_peripheral_vision_range,
                                           num_orientations, agent_initial_pos, agent_initial_angle,
                                           movement_step, self.agent_diameter, obs_resolution))
        self.agent_states = agent_states
        self.elements = config.get_elements_for_agents(agent_pos_list)
        self.predator_diameter = 0.7
        predator_states = []
        predator_pos_list = config.get_initial_predator_positions(agent_pos_list, self.elements)
        num_predators = len(predator_pos_list)
        for i in range(num_predators):
            predator_initial_pos = predator_pos_list[i]
            px, py = predator_initial_pos
            if px < 0 or py < 0 or px >= gw or py >= gh:
                raise Exception(f'Invalid initial predator position: {predator_initial_pos}')
            predator_initial_angle = self.random.uniform(-math.pi, math.pi)
            predator_initial_orientation_index = orientation_helper.get_index_for_angle(predator_initial_angle)
            predator_states.append(PredatorState(self, self.random, i, orientation_helper, num_orientations,
                                                 predator_initial_pos, predator_initial_orientation_index,
                                                 self.predator_diameter, movement_step,
                                                 predator_level))
        self.predator_states = predator_states
        self.update_elements_grid()

    def get_character_states(self) -> List[BaseEnvCharacter]:
        return self.agent_states + self.predator_states

    def grid_cell_contains_character(self, x: int, y: int) -> bool:
        for c_state in self.get_character_states():
            if not c_state.is_done:
                a_x, a_y = tuple(c_state.position)
                radius = c_state.diameter / 2.0
                a_x1, a_y1 = math.floor(a_x - radius), math.floor(a_y - radius)
                if x == a_x1 and y == a_y1:
                    return True
                a_x2, a_y2 = math.floor(a_x + radius), math.floor(a_y + radius)
                if (x == a_x2 and y == a_y2) or (x == a_x1 and y == a_y2) or (x == a_x2 and y == a_y1):
                    return True
        return False

    def attempt_push(self, element: FlatWorldElement, prev_pos: np.ndarray, ax: float, ay: float,
                     ueg=True, allow_removals=True):
        e_x, e_y = element.position
        prev_x, prev_y = tuple(prev_pos)
        diff_x, diff_y = ax - prev_x, ay - prev_y
        if abs(diff_x) > abs(diff_y):
            e_x = e_x + 1 if diff_x > 0 else e_x - 1
        else:
            e_y = e_y + 1 if diff_y > 0 else e_y - 1
        w, h = self.grid_size
        if e_x < 0 or e_y < 0 or e_x >= w or e_y >= h:
            raise PushException()
        if self.grid_cell_contains_character(e_x, e_y):
            raise PushException()
        blocking_element: FlatWorldElement = self.elements_grid[e_y][e_x]
        if blocking_element is not None:
            if blocking_element.is_displaced:
                be_x, be_y = blocking_element.position
                self.attempt_push(blocking_element, np.array(element.position) + 0.5, be_x + 0.5, be_y + 0.5,
                                  ueg=False, allow_removals=blocking_element.is_pushable)
            elif blocking_element.is_destroyed and allow_removals:
                self.elements = [e for e in self.elements if e != blocking_element]
            elif blocking_element.transforms_into_fn and allow_removals:
                new_element = blocking_element.transforms_into_fn(blocking_element.position)
                new_elements = [e for e in self.elements if e != blocking_element and e != element]
                new_elements.append(new_element)
                self.elements = new_elements
            else:
                raise PushException()
        element.position = (e_x, e_y,)
        if ueg:
            self.update_elements_grid()

    def get_touched_elements(self, x1, y1, x2, y2):
        gx1, gy1 = math.floor(x1), math.floor(y1)
        gx2, gy2 = math.floor(x2), math.floor(y2)
        grid = self.elements_grid
        options = [grid[gy1][gx1], grid[gy1][gx2], grid[gy2][gx2], grid[gy2][gx1]]
        return [e for e in set(options) if e is not None]

    def update_elements_grid(self):
        w, h = self.grid_size
        grid = [[None] * w for r in range(h)]
        e: FlatWorldElement
        for e in self.elements:
            x, y = e.position
            grid[y][x] = e
        self.elements_grid = grid

    def remove_elements(self, elements: Iterable[FlatWorldElement]):
        element_set = set(elements)
        self.elements = [e for e in self.elements if e not in element_set]
        self.update_elements_grid()

    def get_default_reward(self) -> float:
        return self.default_reward

    def get_reward_no_action(self) -> float:
        return self.reward_no_action

    def get_reward_toggle(self) -> float:
        return self.reward_toggle

    def get_grid_size(self) -> Tuple[int, int]:
        return self.grid_size

    def get_elements(self):
        return self.elements

    def get_other_entities(self, agent_index: int) -> List[AbstractEnvEntity]:
        return self.elements + self.get_other_agents(agent_index) + self.predator_states

    def get_other_agents(self, agent_index: int) -> List[AbstractEnvEntity]:
        return [a_s for a_s in self.agent_states if a_s.index != agent_index and not a_s.is_done]

    def get_min_cumulative_reward(self):
        return self.min_cumulative_reward

    def get_agent_state(self, index: int):
        return self.agent_states[index]

    def get_closest_agent(self, position: np.ndarray) -> Tuple[AbstractEnvEntity, float, float]:
        min_d_sq = +np.Inf
        closest_agent = None
        closest_diff_x = None
        closest_diff_y = None
        for agent_state in self.agent_states:
            if not agent_state.is_done:
                diff_x, diff_y = tuple(agent_state.position - position)
                d_sq = diff_x ** 2 + diff_y ** 2
                if d_sq < min_d_sq:
                    min_d_sq = d_sq
                    closest_agent = agent_state
                    closest_diff_x, closest_diff_y = diff_x, diff_y
        return closest_agent, closest_diff_x, closest_diff_y

    def collision_with_element_or_another_predator(self, index: int, position: np.ndarray, diameter: float) -> bool:
        x, y = tuple(position)
        r = diameter / (2 * SQRT2)
        x1, y1, x2, y2 = x - r, y - r, x + r, y + r
        g_w, g_h = self.grid_size
        if x1 < 0 or y1 < 0:
            return True
        if x2 >= g_w or y2 >= g_h:
            return True
        touched_elements = self.get_touched_elements(x1, y1, x2, y2)
        if len(touched_elements) > 0:
            return True
        if self.get_colliding_predator(position, diameter, not_pred_index=index):
            return True
        return False

    def get_colliding_predator(self, position: np.ndarray, diameter: float, not_pred_index: int = -1) -> BaseEnvCharacter:
        for predator_state in self.predator_states:
            if predator_state.index != not_pred_index:
                diff_x, diff_y = tuple(predator_state.position - position)
                dist_sq = diff_x ** 2 + diff_y ** 2
                dist_threshold_sq = ((predator_state.diameter + diameter) / 2.0) ** 2
                if dist_sq <= dist_threshold_sq:
                    return predator_state
        return None

    def get_predator_reward(self):
        return self.reward_predator

    def get_observation(self, bkg_color):
        obs_res = self.obs_resolution
        active_predators = [predator_s for predator_s in self.predator_states if not predator_s.is_done]
        num_agents = len(self.agent_states)
        a_positions = np.array([agent_s.position for agent_s in self.agent_states])
        a_pos_x, a_pos_y = a_positions[:, 0], a_positions[:, 1]
        p_positions = np.array([predator_s.position for predator_s in active_predators])
        if len(p_positions) > 0:
            p_pos_x, p_pos_y = p_positions[:, 0], p_positions[:, 1]
        else:
            p_pos_x, p_pos_y = None, None,
        element_corner_1 = np.array([element.position for element in self.elements], dtype=np.float16)
        if len(element_corner_1) > 0:
            e_x1, e_y1 = element_corner_1[:, 0], element_corner_1[:, 1]
            e_x2, e_y2 = e_x1 + 1, e_y1 + 1
        else:
            e_x1, e_y1 = None, None,
            e_x2, e_y2 = None, None,

        # Create agent squares, as seen by other agents
        agent_is_active = np.array([not agent_s.is_done for agent_s in self.agent_states])
        if np.any(agent_is_active):
            agent_radius = self.agent_diameter / 2.0
            active_a_pos_x = a_pos_x[agent_is_active]
            active_a_pos_y = a_pos_y[agent_is_active]
            a_x1, a_y1 = active_a_pos_x - agent_radius, active_a_pos_y - agent_radius
            a_x2, a_y2 = active_a_pos_x + agent_radius, active_a_pos_y + agent_radius
        else:
            a_x1, a_y1 = None, None,
            a_x2, a_y2 = None, None,

        # Create predator squares
        if p_pos_x is not None:
            predator_radius = self.predator_diameter / 2.0
            p_x1, p_y1 = p_pos_x - predator_radius, p_pos_y - predator_radius
            p_x2, p_y2 = p_pos_x + predator_radius, p_pos_y + predator_radius
        else:
            p_x1, p_y1 = None, None,
            p_x2, p_y2 = None, None,

        # Environment bounds
        g_x1, g_y1 = np.array([0]), np.array([0])
        g_w, g_h = self.grid_size
        g_x2, g_y2 = g_x1 + g_w, g_y1 + g_h

        element_colors = np.array([element.color for element in self.elements], dtype=int)
        # TODO Only front colors of agents and predators supported ATM
        agent_colors = np.array([agent_s.get_front_color() for agent_s in self.agent_states], dtype=int)
        active_agent_colors = agent_colors[agent_is_active]
        predator_colors = np.array([predator_s.get_front_color() for predator_s in active_predators], dtype=int)
        cell_sines_cosines = [agent_s.get_cells_sin_cos() for agent_s in self.agent_states]
        cell_sines_list, cell_cosines_list = zip(*cell_sines_cosines)
        cell_sines, cell_cosines = np.array(cell_sines_list), np.array(cell_cosines_list)
        # cell_sines/cell_cosines shape: (n, obs_res)
        bkg_color_np = np.array(bkg_color, dtype=int)[:3]

        rep_a_pos_x = np.repeat(a_pos_x, obs_res, axis=0)
        rep_a_pos_y = np.repeat(a_pos_y, obs_res, axis=0)
        # rep_a_pos_x and rep_a_pos_y shape: (n * obs_res,)
        flat_cell_sines = cell_sines.reshape((num_agents * obs_res,))
        flat_cell_cosines = cell_cosines.reshape((num_agents * obs_res,))

        ent_color_list = []
        ent_x1_list = []
        ent_y1_list = []
        ent_x2_list = []
        ent_y2_list = []
        # Combined entity squares and colors
        if a_x1 is not None:
            ent_x1_list.append(a_x1)
            ent_y1_list.append(a_y1)
            ent_x2_list.append(a_x2)
            ent_y2_list.append(a_y2)
            ent_color_list.append(active_agent_colors)
        if e_x1 is not None:
            ent_x1_list.append(e_x1)
            ent_y1_list.append(e_y1)
            ent_x2_list.append(e_x2)
            ent_y2_list.append(e_y2)
            ent_color_list.append(element_colors)
        if p_x1 is not None:
            ent_x1_list.append(p_x1)
            ent_y1_list.append(p_y1)
            ent_x2_list.append(p_x2)
            ent_y2_list.append(p_y2)
            ent_color_list.append(predator_colors)
        if len(ent_color_list) > 0:
            ent_colors = np.concatenate(ent_color_list)
            ent_x1 = np.concatenate(ent_x1_list)
            ent_y1 = np.concatenate(ent_y1_list)
            ent_x2 = np.concatenate(ent_x2_list)
            ent_y2 = np.concatenate(ent_y2_list)
        else:
            ent_colors = None
            ent_x1, ent_y1, ent_x2, ent_y2 = None, None, None, None,
        with np.errstate(divide='ignore', invalid='ignore'):
            if ent_x1 is not None:
                ent_distances = min_distances_to_rects(rep_a_pos_x, rep_a_pos_y,
                                                       flat_cell_sines, flat_cell_cosines,
                                                       ent_x1, ent_y1, ent_x2, ent_y2,
                                                       check_direction=True, inside_ok=False)
            else:
                ent_distances = None
            g_distances = min_distances_to_rects(rep_a_pos_x, rep_a_pos_y,
                                                 flat_cell_sines, flat_cell_cosines,
                                                 g_x1, g_y1, g_x2, g_y2,
                                                 check_direction=True, inside_ok=True)
            cell_channels = self.get_channels(ent_distances, g_distances, ent_colors, bkg_color_np)
            # cell_channels shape: (n * obs_res, 4,)
            batch_obs = cell_channels.reshape((num_agents, obs_res, 4,))
            batch_obs = np.transpose(batch_obs, axes=(0, 2, 1,))
        if self.squeeze and len(self.agent_states) == 1:
            batch_obs = batch_obs.squeeze(0)
        return batch_obs

    def get_channels(self, ent_distances: np.ndarray, g_distances: np.ndarray, ent_colors: np.ndarray,
                     bkg_color: np.ndarray, k_distance=5.0):
        # Given:
        #   m entities (elements + agents + predators)
        #   n agents
        #   k visual resolution
        # Shapes:
        #   ent_distances: (m, n * k,)
        #   ent_colors: (m, 4,)
        #   g_distances: (1, n * k,)
        num_cells = g_distances.shape[1]
        wall_depths = g_distances * 255.0 / (g_distances + k_distance)
        default_channels = np.empty((num_cells, 4), dtype=int)
        default_channels[:, :3] = bkg_color
        default_channels[:, 3] = wall_depths
        if ent_distances is not None:
            channels = self.get_ent_colors_depths(ent_distances, ent_colors, default_channels, k_distance)
        else:
            channels = default_channels
        return channels

    def get_ent_colors_depths(self, ent_distances: np.ndarray, ent_colors: np.ndarray, default_channels: np.ndarray,
                              k_distance: float):
        # Given:
        #   m entities (elements + agents + predators)
        #   n agents
        #   k visual resolution
        # Shapes:
        #   ent_distances: (m, n * k,)
        #   ent_colors: (m, 4,)
        num_cells = ent_distances.shape[1]
        min_indexes = np.argmin(ent_distances, 0)
        # min_indexes: (n * k,) | values are entity indexes
        num_cell_range = np.arange(0, num_cells)
        selected_distances = ent_distances[min_indexes, num_cell_range]
        # selected_distances: (n * k,)
        invalid_distances = np.isinf(selected_distances)
        # invalid_distances: (n * k,)
        # Note that numpy slices are references
        selected_colors = np.copy(ent_colors[min_indexes, :])
        # selected_colors: (n * k, 4,)
        selected_alphas = np.copy(selected_colors[:, 3])
        # selected_alphas: (n * k,)
        selected_colors[:, 3] = selected_distances * 255.0 / (selected_distances + k_distance)
        selected_colors[invalid_distances, :] = default_channels[invalid_distances, :]
        selected_translucent = (selected_alphas < 254) & ~invalid_distances
        if np.any(selected_translucent):
            sub_ent_distances = np.copy(ent_distances)
            # Clear prev minimums
            sub_ent_distances[min_indexes, num_cell_range] = np.Inf
            # Keep only translucent cells
            sub_ent_distances = sub_ent_distances[:, selected_translucent]
            # Keep only default channels of translucent cells
            sub_default_channels = default_channels[selected_translucent, :]
            # Recursive call
            sub_selected_colors = self.get_ent_colors_depths(sub_ent_distances, ent_colors, sub_default_channels,
                                                             k_distance)
            # sub_selected_colors: (n * k * fraction, 4,)
            sub_selected_alphas = selected_alphas[selected_translucent]
            # sub_selected_alphas: (n * k * fraction,)
            # Update colors of translucent cells
            prior_weight = (sub_selected_alphas / 255.0)[:, None]
            selected_colors[selected_translucent, :3] = \
                selected_colors[selected_translucent, :3] * prior_weight + \
                sub_selected_colors[:, :3] * (1 - prior_weight)
        return selected_colors

    def apply_predator_actions(self):
        for p_state in self.predator_states:
            p_state.apply_stochastic_action()

    def apply_actions(self, actions: _multi_action_type):
        self.apply_predator_actions()
        if not self.is_batched:
            actions = [actions]
        agent_states = self.agent_states
        n = len(agent_states)
        if n != len(actions):
            raise Exception(f'Expected {n} actions, but got {len(actions)}')
        a_indexes = range(n)
        if n > 0:
            a_indexes = list(a_indexes)
            self.random.shuffle(a_indexes)
            mapped_rewards = np.array([agent_states[i].apply_action(actions[i]) for i in a_indexes])
            rewards = np.zeros((n,))
            rewards[a_indexes] = mapped_rewards
        else:
            rewards = np.array([agent_states[0].apply_action(actions[0])])
        if self.squeeze and n == 1:
            rewards = rewards.item()
        return rewards

    def get_infos(self):
        agent_states = self.agent_states
        infos = [agent_s.get_info() for agent_s in agent_states]
        if self.squeeze and len(infos) == 1:
            infos = infos[0]
        return infos

    def get_done_states(self):
        agent_states = self.agent_states
        n = len(agent_states)
        done_s = np.array([agent_states[i].is_done for i in range(n)], dtype=bool)
        if self.squeeze and n == 1:
            done_s = done_s.item()
        return done_s

