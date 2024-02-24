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


class Metalearning2Task(BaseMetalearningTask):
    def __init__(self, rigid_seed: int, mtt: MetalearningTaskType, grid_size: Tuple[int, int],
                 final_food_type: FinalFoodType, num_food_elements: int,
                 element_fns: list, element_positions: List[Tuple[int, int]],
                 left_to_x: List[int], right_from_x: List[int],
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
        agent_positions = []
        food_positions = []
        final_food_positions = []
        for y in range(gh):
            to_x = left_to_x[y]
            from_x = right_from_x[y]
            row_agent_pos = [(x, y) for x in range(to_x)]
            agent_positions.extend(row_agent_pos)
            row_food_pos = [(x, y) for x in range(from_x, gw)]
            food_positions.extend(row_food_pos)
            row_final_food_pos = [(x, y) for x in range(from_x + 1, gw)]
            final_food_positions.extend(row_final_food_pos)
        self.agent_pos_set = set(agent_positions).difference(element_pos_set)
        self.agent_position_pool = list(self.agent_pos_set)
        self.food_pos_set = set(food_positions)
        self.final_food_pos_set = set(final_food_positions)

    def get_min_cumulative_reward(self):
        return self.min_cumulative_reward

    @staticmethod
    def generate(grid_size: Tuple[int, int], mmt: MetalearningTaskType, final_food_type: FinalFoodType,
                 num_food_elements: int, min_cumulative_reward: float, difficulty: float, seed=None):
        rs = RandomState()
        rs.seed(seed)
        is_dependent = mmt == MetalearningTaskType.DEPENDENT
        element_fns, element_positions, left_to_x, right_from_x = \
            Metalearning2Task.generate_elements(rs, grid_size, is_dependent, difficulty)
        rigid_seed = rs.randint(0, 1000000)
        return Metalearning2Task(rigid_seed, mmt, grid_size, final_food_type, num_food_elements, element_fns,
                                 element_positions, left_to_x, right_from_x, min_cumulative_reward)

    @staticmethod
    def generate_elements(rs: RandomState, grid_size: Tuple[int, int], is_dependent: bool,
                          difficulty: float, p_dependent=0.1, p_extra_box=0.2):
        gw, gh = grid_size
        buffer_cols = 2
        if gw < buffer_cols * 2 + 1 or gh < 3:
            raise Exception(f'Grid size too small: {grid_size}')
        grid = [[False] * gw for _ in range(gh)]
        elements = []
        positions = []
        min_x = buffer_cols
        max_x = gw - buffer_cols - 2
        x_start = rs.randint(min_x, max_x + 1)
        y_start = 0
        pos = x_start, y_start,
        direction = 0, 1,
        dir_count = 0
        element_type: EnvElement = Metalearning2Task.get_start_element_type(rs)
        while pos[1] < gh:
            if is_dependent and rs.uniform() < p_dependent:
                element_fn = Metalearning2Task.get_dependent_element_fn(rs, element_type)
            else:
                element_fn = Metalearning2Task.get_element_fn(element_type)
            assert element_fn is not None
            elements.append(element_fn)
            positions.append(pos)
            x, y = pos
            grid[y][x] = True
            pos, ok = Metalearning2Task.next_position(pos, direction, min_x, max_x, grid)
            if not ok:
                dir_count = 0
                direction = 0, 1,
            else:
                dir_count += 1
            if dir_count >= 4:
                new_dir = Metalearning2Task.new_direction(rs)
                if new_dir != direction:
                    dir_count = 0
                    direction = new_dir
            element_type = Metalearning2Task.next_element_type(rs, element_type)
        elements, positions, left_to_x, right_from_x = \
            Metalearning2Task.finalize_elements(rs, elements, positions, grid_size)
        if difficulty < 1.0:
            indexes = set([idx for idx in range(len(elements)) if rs.uniform() < difficulty])
            elements = [e for i, e in enumerate(elements) if i in indexes]
            positions = [p for i, p in enumerate(positions) if i in indexes]
        if rs.uniform() < p_extra_box:
            for attempt in range(20):
                box_y = rs.randint(1, gh - 1)
                row_w = left_to_x[box_y]
                min_box_x = max(0, row_w - 2)
                box_x = rs.randint(min_box_x, row_w)
                box_pos = box_x, box_y,
                mid_x, mid_y = box_pos
                occupied = False
                for x in range(mid_x - 1, mid_x + 2):
                    for y in range(mid_y - 1, mid_y + 2):
                        if grid[y][x]:
                            occupied = True
                            break
                if not occupied:
                    elements.append(FlatWorldElement.box)
                    positions.append(box_pos)
                    grid[mid_y][mid_x] = True
                    break
        return elements, positions, left_to_x, right_from_x,

    @staticmethod
    def finalize_elements(rs: RandomState, elements: list, positions: List[Tuple[int, int]],
                          grid_size: Tuple[int, int],
                          box_prob=0.2):
        span = 2
        gap_candidate_indexes = []
        for i in range(span, len(positions) - span):
            prev_x = None
            prev_y = None
            x_changed = False
            y_changed = False
            for j in range(i - span, i + span + 1):
                x, y = positions[j]
                if prev_x is not None and prev_x != x:
                    x_changed = True
                if prev_y is not None and prev_y != y:
                    y_changed = True
                prev_x, prev_y = x, y
            if not x_changed or not y_changed:
                gap_candidate_indexes.append(i)
        assert len(gap_candidate_indexes) > 0, f'Bad positions: {positions}'
        groups = itertools.groupby(positions, key=lambda _p: _p[1])
        gw, gh = grid_size
        left_to_x = [0] * gh
        right_from_x = [0] * gh
        for gy, gpos in groups:
            gpos_list = list(gpos)
            max_pos = max(gpos_list, key=lambda _p: _p[0])
            min_pos = min(gpos_list, key=lambda _p: _p[0])
            max_x, min_x = max_pos[0], min_pos[0]
            left_to_x[gy] = min_x
            right_from_x[gy] = max_x + 1
        new_elements = copy.deepcopy(elements)
        new_positions = copy.deepcopy(positions)
        gap_index = rs.choice(gap_candidate_indexes)
        if rs.uniform() < box_prob:
            new_elements[gap_index] = FlatWorldElement.box
        else:
            new_elements.pop(gap_index)
            new_positions.pop(gap_index)
        return new_elements, new_positions, left_to_x, right_from_x

    @staticmethod
    def is_edge_position(pos: Tuple[int, int], gw: int, gh: int):
        x, y = pos
        return x == 0 or y == 0 or x == gw - 1 or y == gh - 1

    @staticmethod
    def next_element_type(rs: RandomState, element_type: EnvElement, p_change=0.2):
        if element_type is None or rs.uniform() < p_change:
            return Metalearning2Task.get_random_element_type(rs)
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
        alt_element_type = Metalearning2Task.get_alt_element_type(rs, element_type)
        alt_fn = alt_element_type.get_constructor_fn()

        def _de_fn(pos: Tuple[int, int]):
            return nominal_fn(pos) if rs.uniform() < 0.5 else alt_fn(pos)

        return _de_fn

    @staticmethod
    def get_alt_element_type(rs: RandomState, element_type: EnvElement):
        aet = None
        for attempt in range(20):
            aet = Metalearning2Task.get_random_element_type(rs)
            if aet != element_type:
                break
        return aet

    @staticmethod
    def get_start_element_type(rs: RandomState):
        return Metalearning2Task.get_random_element_type(rs)

    @staticmethod
    def get_random_element_type(rs: RandomState, weights=None):
        if weights is None:
            weights = [3, 1, 1, 2]
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
        return Metalearning2Task.pos_choice(rs, self.agent_position_pool, num_agents)

    def get_elements_for_agents(self, initial_agent_positions: List[Tuple[int, int]]) -> List[FlatWorldElement]:
        base_elements = [fn(pos) for fn, pos in zip(self.element_fns, self.element_positions)]
        ff_rs = self.get_random_state(base_elements=base_elements, offset=1003)
        elements = []
        elements.extend(base_elements)
        remaining_pos_set = self.food_pos_set
        food_pos = Metalearning2Task.pos_choice(ff_rs, list(remaining_pos_set), self.num_food_elements)
        for pos in food_pos:
            elements.append(FlatWorldElement.food(pos))
        fft = self.final_food_type
        num_ff = len(initial_agent_positions) * fft.get_num_elements()
        remaining_pos_set = self.final_food_pos_set.difference(set(food_pos))
        ff_pos = Metalearning2Task.pos_choice(ff_rs, list(remaining_pos_set), num_ff)
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
                h = sum(Metalearning2Task.element_hash(element) * (idx % 16) for idx, element in enumerate(base_elements))
                rs_seed = self.rigid_seed + (h % 100000000) + offset
                return RandomState(rs_seed)
            else:
                return RandomState(self.rigid_seed + offset)
        else:
            return self.random

