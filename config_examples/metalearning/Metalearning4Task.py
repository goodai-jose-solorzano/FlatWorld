import copy
import itertools
import math
import numpy as np
from typing import List, Tuple, Union
from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement
from flat_world.FlatWorldMultiAgentConfig import FlatWorldMultiAgentConfig
from numpy.random import RandomState

from tasks.EnvElement import EnvElement
from tasks.FinalFoodType import FinalFoodType
from tasks.metalearning.BaseMetalearningTask import BaseMetalearningTask
from tasks.metalearning.MetalearningTaskType import MetalearningTaskType


class Metalearning4Task(BaseMetalearningTask):
    def __init__(self, rigid_seed: int, mtt: MetalearningTaskType, grid_size: Tuple[int, int],
                 final_food_type: FinalFoodType, num_food_elements: int,
                 element_fns: list, element_positions: List[Tuple[int, int]],
                 box_from_x: int, box_to_x: int, box_from_y: int, box_to_y: int,
                 min_cumulative_reward: float):
        # Call generate() to create tasks
        super().__init__()
        self.grid_size = grid_size
        self.rigid_seed = rigid_seed
        self.task_type = mtt
        self.final_food_type = final_food_type
        self.num_food_elements = num_food_elements
        self.element_fns = element_fns
        self.element_positions = element_positions
        self.min_cumulative_reward = min_cumulative_reward
        element_pos_set = set(element_positions)
        gw, gh = grid_size
        all_positions = [(x, y) for x in range(gw) for y in range(gh)]
        food_positions = [(x, y) for x in range(box_from_x + 1, box_to_x - 1)
                          for y in range(box_from_y + 1, box_to_y - 1)]
        self.agent_pos_set = set(all_positions).difference(set(food_positions)).difference(element_pos_set)
        self.agent_position_pool = list(self.agent_pos_set)
        self.food_pos_set = set(food_positions)

    def get_min_cumulative_reward(self):
        return self.min_cumulative_reward

    @staticmethod
    def generate(grid_size: Tuple[int, int], mmt: MetalearningTaskType, final_food_type: FinalFoodType,
                 num_food_elements: int, min_cumulative_reward: float, difficulty: float, seed=None):
        rs = RandomState()
        rs.seed(seed)
        is_dependent = mmt == MetalearningTaskType.DEPENDENT
        element_fns, element_positions, box_bounds = \
            Metalearning4Task.generate_elements(rs, grid_size, num_food_elements, is_dependent, difficulty)
        rigid_seed = rs.randint(0, 1000000)
        b1, b2, b3, b4 = box_bounds
        return Metalearning4Task(rigid_seed, mmt, grid_size, final_food_type, num_food_elements, element_fns,
                                 element_positions, b1, b2, b3, b4, min_cumulative_reward)

    @staticmethod
    def generate_elements(rs: RandomState, grid_size: Tuple[int, int], num_food_elements: int,
                          is_dependent: bool, difficulty: float, p_dependent=0.1,
                          p_extra_box=0.25) -> Tuple[list, list, Tuple[int, int, int, int]]:
        gw, gh = grid_size
        min_pen_size = math.ceil(math.sqrt(num_food_elements + 1) * 2)
        min_arena_size = min_pen_size + 3
        if gw < min_arena_size or gh < min_arena_size:
            raise Exception(f'Arena size is too small: {grid_size}')
        min_box_size = min_pen_size + 2
        from_x = rs.randint(2, gw - 1 - min_box_size)
        from_y = rs.randint(2, gh - 1 - min_box_size)
        to_x = rs.randint(from_x + min_pen_size + 1, gw - 2)
        to_y = rs.randint(from_y + min_pen_size + 1, gh - 2)
        bounds = from_x, to_x, from_y, to_y,
        elements = []
        positions = []
        element_type: EnvElement = Metalearning4Task.get_start_element_type(rs)
        top_x_pos = [(x, from_y) for x in range(from_x, to_x)]
        bottom_x_pos = [(x, to_y - 1) for x in range(to_x - 1, from_x - 1, -1)]
        left_y_pos = [(from_x, y) for y in range(to_y - 2, from_y, -1)]
        right_y_pos = [(to_x - 1, y) for y in range(from_y + 1, to_y - 1)]
        pos_sequence = top_x_pos + right_y_pos + bottom_x_pos + left_y_pos
        for pos in pos_sequence:
            if is_dependent and rs.uniform() < p_dependent:
                element_fn = Metalearning4Task.get_dependent_element_fn(rs, element_type)
            else:
                element_fn = Metalearning4Task.get_element_fn(element_type)
            assert element_fn is not None
            elements.append(element_fn)
            positions.append(pos)
            element_type = Metalearning4Task.next_element_type(rs, element_type)
        elements, positions = \
            Metalearning4Task.finalize_elements(rs, elements, positions)
        if difficulty < 1.0:
            indexes = set([idx for idx in range(len(elements)) if rs.uniform() < difficulty])
            elements = [e for i, e in enumerate(elements) if i in indexes]
            positions = [p for i, p in enumerate(positions) if i in indexes]
        if rs.uniform() < p_extra_box:
            for attempt in range(20):
                box_y = rs.randint(1, gh - 1)
                box_x = rs.randint(1, gw - 1)
                if from_x <= box_x < to_x and from_y <= box_y < to_y:
                    continue
                box_element = FlatWorldElement.box
                elements.append(box_element)
                positions.append((box_x, box_y))
                break
        return elements, positions, bounds,

    @staticmethod
    def finalize_elements(rs: RandomState, elements: list, positions: List[Tuple[int, int]], box_prob=0.25):
        new_elements = list(elements)
        new_positions = list(positions)
        gap_index = rs.randint(0, len(positions))
        if rs.uniform() < box_prob:
            new_elements[gap_index] = FlatWorldElement.box
        else:
            new_elements.pop(gap_index)
            new_positions.pop(gap_index)
        return new_elements, new_positions,

    @staticmethod
    def is_edge_position(pos: Tuple[int, int], gw: int, gh: int):
        x, y = pos
        return x == 0 or y == 0 or x == gw - 1 or y == gh - 1

    @staticmethod
    def next_element_type(rs: RandomState, element_type: EnvElement, p_change=0.2):
        if element_type is None or rs.uniform() < p_change:
            return Metalearning4Task.get_random_element_type(rs)
        return element_type

    @staticmethod
    def new_direction(rs: RandomState, p_sideways=0.5) -> Tuple[int, int]:
        if rs.uniform() < p_sideways:
            if rs.uniform() < 0.5:
                return +1, 0,
            else:
                return -1, 0,
        else:
            return 0, +1,

    @staticmethod
    def next_position(pos: Tuple[int, int], direction: Tuple[int, int],
                      min_x: int, max_x: int, grid: List[list]) -> Tuple[Tuple[int, int], bool]:
        x, y = pos
        dx, dy = direction
        new_x, new_y = x + dx, y + dy,
        if min_x <= new_x <= max_x and new_y < len(grid) and not grid[new_y][new_x]:
            if dx == 0 or new_y == 0 or not grid[new_y - 1][new_x]:
                return (new_x, new_y), True,
        return (x, y + 1), False,

    @staticmethod
    def is_invalid_pos(pos: Tuple[int, int], grid: List[list]):
        x, y = pos
        if y < 0 or x < 0 or y >= len(grid):
            return True
        row = grid[y]
        if x >= len(row):
            return True
        return row[x]

    @staticmethod
    def get_element_fn(element_type: EnvElement):
        return element_type.get_constructor_fn()

    @staticmethod
    def get_dependent_element_fn(rs: RandomState, element_type: EnvElement):
        nominal_fn = element_type.get_constructor_fn()
        alt_element_type = Metalearning4Task.get_alt_element_type(rs, element_type)
        alt_fn = alt_element_type.get_constructor_fn()

        def _de_fn(pos: Tuple[int, int]):
            return nominal_fn(pos) if rs.uniform() < 0.5 else alt_fn(pos)

        return _de_fn

    @staticmethod
    def get_alt_element_type(rs: RandomState, element_type: EnvElement):
        aet = None
        for attempt in range(20):
            aet = Metalearning4Task.get_random_element_type(rs)
            if aet != element_type:
                break
        return aet

    @staticmethod
    def get_start_element_type(rs: RandomState):
        return Metalearning4Task.get_random_element_type(rs)

    @staticmethod
    def get_random_element_type(rs: RandomState, weights=None):
        if weights is None:
            weights = [2, 1, 1, 3]
        element_types = [EnvElement.WALL, EnvElement.FENCE, EnvElement.BOX, EnvElement.TRANSLUCENT]
        weights = np.array(weights)
        p = weights / np.sum(weights)
        return rs.choice(element_types, p=p)

    @staticmethod
    def element_hash(element: FlatWorldElement) -> int:
        r, g, b, a = element.color
        return a + b << 8 + g << 16 + r << 24

    @staticmethod
    def pos_choice(rs: RandomState, pos_list: List[Tuple[int, int]], size: int):
        pos_idx = rs.choice(range(len(pos_list)), size=size, replace=False)
        selected = np.array(pos_list)[pos_idx]
        return [tuple(pos) for pos in selected]

    def get_initial_agent_angles(self, num_agents: int) -> Union[List[float], np.ndarray]:
        rs = self.get_random_state(offset=1001)
        return rs.uniform(0, 2 * math.pi, size=num_agents)

    def get_initial_agent_positions(self, num_agents: int) -> List[Tuple[int, int]]:
        rs = self.get_random_state(offset=1002)
        return Metalearning4Task.pos_choice(rs, self.agent_position_pool, num_agents)

    def get_elements_for_agents(self, initial_agent_positions: List[Tuple[int, int]]) -> List[FlatWorldElement]:
        base_elements = [fn(pos) for fn, pos in zip(self.element_fns, self.element_positions)]
        ff_rs = self.get_random_state(base_elements=base_elements, offset=1003)
        elements = []
        elements.extend(base_elements)
        remaining_pos_set = self.food_pos_set
        food_pos = Metalearning4Task.pos_choice(ff_rs, list(remaining_pos_set), self.num_food_elements)
        for pos in food_pos:
            elements.append(FlatWorldElement.food(pos))
        fft = self.final_food_type
        num_ff = len(initial_agent_positions) * fft.get_num_elements()
        remaining_pos_set = self.food_pos_set.difference(set(food_pos))
        rps_list = list(remaining_pos_set)
        if len(rps_list) >= num_ff:
            ff_pos = Metalearning4Task.pos_choice(ff_rs, rps_list, num_ff)
            ff_elem_fn = fft.get_element_fns()
            for pos_idx, pos in enumerate(ff_pos):
                fn = ff_elem_fn[pos_idx % len(ff_elem_fn)]
                elements.append(fn(pos))
        return elements

    def get_grid_size(self) -> Tuple[int, int]:
        return self.grid_size

    def get_reward_upper_bound(self):
        r = 0.5 if self.final_food_type == FinalFoodType.BLUE else 1.0
        r += self.num_food_elements * 1.0
        return r

    def get_random_state(self, offset=0, base_elements=None):
        if self.task_type == MetalearningTaskType.RIGID:
            return RandomState(self.rigid_seed + offset)
        elif self.task_type == MetalearningTaskType.DEPENDENT:
            if base_elements is not None:
                h = sum(Metalearning4Task.element_hash(element) * (idx % 16) for idx, element in enumerate(base_elements))
                rs_seed = self.rigid_seed + (h % 100000000) + offset
                return RandomState(rs_seed)
            else:
                return RandomState(self.rigid_seed + offset)
        else:
            return self.random
