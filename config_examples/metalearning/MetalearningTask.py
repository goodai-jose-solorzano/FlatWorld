import copy
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


class MetalearningTask(BaseMetalearningTask):
    def __init__(self, rigid_seed: int, mtt: MetalearningTaskType, grid_size: Tuple[int, int],
                 final_food_type: FinalFoodType, num_food_elements: int,
                 element_fns: list, element_positions: List[Tuple[int, int]],
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
        all_pos_set = set([(x, y) for x in range(gw) for y in range(gh)])
        self.avail_pos_set = all_pos_set.difference(element_pos_set)
        self.available_positions = list(self.avail_pos_set)

    def get_min_cumulative_reward(self):
        return self.min_cumulative_reward

    @staticmethod
    def generate(grid_size: Tuple[int, int], mmt: MetalearningTaskType, final_food_type: FinalFoodType,
                 num_food_elements: int, min_brick_elements: int, max_brick_elements: int,
                 min_cumulative_reward: float, seed=None):
        rs = RandomState()
        rs.seed(seed)
        num_elements = rs.randint(min_brick_elements, max_brick_elements + 1)
        is_dependent = mmt == MetalearningTaskType.DEPENDENT
        element_fns, element_positions = MetalearningTask.generate_elements(rs, grid_size, num_elements, is_dependent)
        rigid_seed = rs.randint(0, 1000000)
        return MetalearningTask(rigid_seed, mmt, grid_size, final_food_type, num_food_elements, element_fns,
                                element_positions, min_cumulative_reward)

    @staticmethod
    def generate_elements(rs: RandomState, grid_size: Tuple[int, int], num_elements: int, is_dependent: bool,
                          p_dependent=0.1, p_extra_box=0.1):
        gw, gh = grid_size
        if gw < 3 or gh < 3:
            raise Exception(f'Grid size too small: {grid_size}')
        grid = [[False] * gw for _ in range(gh)]
        elements = []
        positions = []
        pos, direction = MetalearningTask.get_start_pos(rs, gw, gh, grid)
        element_type: EnvElement = MetalearningTask.get_start_element_type(rs)
        has_gap_or_box = element_type == EnvElement.BOX
        while len(elements) < num_elements:
            if is_dependent and rs.uniform() < p_dependent:
                element_fn = MetalearningTask.get_dependent_element_fn(rs, element_type)
            else:
                element_fn = MetalearningTask.get_element_fn(element_type)
            if element_fn is not None:
                if element_fn == FlatWorldElement.box:
                    has_gap_or_box = True
                elements.append(element_fn)
                positions.append(pos)
                x, y = pos
                grid[y][x] = True
            else:
                has_gap_or_box = True
            direction = MetalearningTask.new_direction(rs, pos, direction, grid, has_gap_or_box)
            pos, new_dir, is_new_start = MetalearningTask.next_position(rs, pos, direction, grid)
            if is_new_start:
                assert new_dir is not None
                direction = new_dir
                element_type: EnvElement = MetalearningTask.get_start_element_type(rs)
                has_gap_or_box = element_type == EnvElement.BOX or not MetalearningTask.is_edge_position(pos, gw, gh)
            else:
                element_type = MetalearningTask.next_element_type(rs, element_type)
        if rs.uniform() < p_extra_box:
            for attempt in range(20):
                box_pos = rs.randint(1, gw - 1), rs.randint(1, gh - 1),
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
        return elements, positions,

    @staticmethod
    def is_edge_position(pos: Tuple[int, int], gw: int, gh: int):
        x, y = pos
        return x == 0 or y == 0 or x == gw - 1 or y == gh - 1

    @staticmethod
    def next_element_type(rs: RandomState, element_type: EnvElement, p_change=0.2, p_gap=0.05):
        if element_type is None or rs.uniform() < p_change:
            return MetalearningTask.get_random_element_type(rs)
        if rs.uniform() < p_gap:
            return None
        return element_type

    @staticmethod
    def next_position(rs: RandomState, pos: Tuple[int, int], direction: Tuple[int, int], grid: List[list],
                      p_start_over=0.05):
        dir_np = np.array(direction)
        next_pos = tuple(np.array(pos) + dir_np)
        if rs.uniform() < p_start_over or not MetalearningTask.is_open_position(next_pos, pos, grid):
            gh = len(grid)
            gw = len(grid[0])
            new_pos, new_dir = MetalearningTask.get_start_pos(rs, gw, gh, grid)
            return new_pos, new_dir, True,
        else:
            return next_pos, None, False,

    @staticmethod
    def is_open_position(pos: Tuple[int, int], origin: Tuple[int, int], grid: List[list]):
        mid_x, mid_y = pos
        o_x, o_y = origin
        if mid_x == o_x:
            # Vertical move
            for x in range(mid_x - 1, mid_x + 2):
                for y in range(mid_y - 1, mid_y + 2):
                    if y != o_y:
                        if MetalearningTask.is_invalid_pos((x, y), grid):
                            return False
        elif mid_y == o_y:
            # Horizontal move
            for x in range(mid_x - 1, mid_x + 2):
                for y in range(mid_y - 1, mid_y + 2):
                    if x != o_x:
                        if MetalearningTask.is_invalid_pos((x, y), grid):
                            return False
        return True

    @staticmethod
    def new_direction(rs: RandomState, pos: Tuple[int, int], direction: Tuple[int, int], grid: List[list],
                      has_gap_or_box: bool, p_change=0.2):
        if rs.uniform() < p_change or (MetalearningTask.direction_blocked(pos, direction, grid) and not has_gap_or_box):
            dx, dy = direction
            alt_dirs = [(-1, 0), (+1, 0)] if dx == 0 else [(0, -1), (0, +1)]
            if MetalearningTask.direction_blocked(pos, alt_dirs[0], grid):
                return alt_dirs[1]
            elif MetalearningTask.direction_blocked(pos, alt_dirs[1], grid):
                return alt_dirs[0]
            else:
                alt_dir_idx = rs.randint(0, len(alt_dirs))
                return alt_dirs[alt_dir_idx]
        else:
            return direction

    @staticmethod
    def direction_blocked(pos: Tuple[int, int], direction: Tuple[int, int], grid: List[list]) -> bool:
        dir_np = np.array(direction)
        next_pos = np.array(pos) + dir_np
        next_next_pos = next_pos + dir_np
        return MetalearningTask.is_invalid_pos(tuple(next_pos), grid) or \
            MetalearningTask.is_invalid_pos(tuple(next_next_pos), grid)

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
        return None if element_type is None else element_type.get_constructor_fn()

    @staticmethod
    def get_dependent_element_fn(rs: RandomState, element_type: EnvElement):
        if element_type is None:
            return None
        nominal_fn = element_type.get_constructor_fn()
        alt_element_type = MetalearningTask.get_alt_element_type(rs, element_type)
        alt_fn = alt_element_type.get_constructor_fn()

        def _de_fn(pos: Tuple[int, int]):
            return nominal_fn(pos) if rs.uniform() < 0.5 else alt_fn(pos)

        return _de_fn

    @staticmethod
    def get_alt_element_type(rs: RandomState, element_type: EnvElement):
        aet = None
        for attempt in range(20):
            aet = MetalearningTask.get_random_element_type(rs)
            if aet != element_type:
                break
        return aet

    @staticmethod
    def get_start_element_type(rs: RandomState):
        return MetalearningTask.get_random_element_type(rs)

    @staticmethod
    def get_random_element_type(rs: RandomState, weights=None):
        if weights is None:
            weights = [3, 1, 1, 2]
        element_types = [EnvElement.WALL, EnvElement.FENCE, EnvElement.BOX, EnvElement.TRANSLUCENT]
        weights = np.array(weights)
        p = weights / np.sum(weights)
        return rs.choice(element_types, p=p)

    @staticmethod
    def get_non_edge_start_pos(rs: RandomState, gw: int, gh: int, grid: List[list]):
        mid_x = None
        mid_y = None
        direction = None
        directions = [(0, +1), (0, -1), (-1, 0), (+1, 0)]
        len_dir = len(directions)
        for attempt in range(20):
            mid_x = rs.randint(1, gw - 1)
            mid_y = rs.randint(1, gh - 1)
            invalid = False
            for x in range(mid_x - 1, mid_x + 2):
                for y in range(mid_y - 1, mid_y + 2):
                    if MetalearningTask.is_invalid_pos((x, y), grid):
                        invalid = True
                        break
            if not invalid:
                dir_idx = rs.randint(0, len_dir)
                direction = directions[dir_idx]
                if not MetalearningTask.direction_blocked((mid_x, mid_y), direction, grid):
                    break
        if direction is None:
            direction = 0, 1,
        return (mid_x, mid_y,), direction,

    @staticmethod
    def get_start_pos(rs: RandomState, gw: int, gh: int, grid: List[list], p_edge_start=0.80):
        if rs.uniform() >= p_edge_start:
            return MetalearningTask.get_non_edge_start_pos(rs, gw, gh, grid)
        x = None
        y = None
        direction = None
        for attempt in range(20):
            if rs.uniform() < 0.5:
                x = rs.randint(1, gw - 1)
                y = rs.choice([0, gh - 1])
                direction = (0, 1 if y == 0 else -1,)
            else:
                x = rs.choice([0, gw - 1])
                y = rs.randint(1, gh - 1)
                direction = (1 if x == 0 else -1, 0,)
            if not MetalearningTask.is_invalid_pos((x, y), grid) and \
                    not MetalearningTask.direction_blocked((x, y), direction, grid):
                break
        return (x, y,), direction,

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
        return MetalearningTask.pos_choice(rs, self.available_positions, num_agents)

    def get_elements_for_agents(self, initial_agent_positions: List[Tuple[int, int]]) -> List[FlatWorldElement]:
        base_elements = [fn(pos) for fn, pos in zip(self.element_fns, self.element_positions)]
        elements = []
        elements.extend(base_elements)
        remaining_pos_set = self.avail_pos_set.difference(set(initial_agent_positions))
        rs = self.get_random_state(offset=1004)
        food_pos = MetalearningTask.pos_choice(rs, list(remaining_pos_set), self.num_food_elements)
        for pos in food_pos:
            elements.append(FlatWorldElement.food(pos))
        ff_rs = self.get_random_state(base_elements=base_elements, offset=1003)
        fft = self.final_food_type
        num_ff = len(initial_agent_positions) * fft.get_num_elements()
        remaining_pos_set = remaining_pos_set.difference(set(food_pos))
        ff_pos = MetalearningTask.pos_choice(ff_rs, list(remaining_pos_set), num_ff)
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
                h = sum(MetalearningTask.element_hash(element) * (idx % 16) for idx, element in enumerate(base_elements))
                rs_seed = self.rigid_seed + (h % 100000000) + offset
                return RandomState(rs_seed)
            else:
                return RandomState(self.rigid_seed + offset)
        else:
            return self.random

