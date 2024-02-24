import copy
import math
import numpy as np
from typing import List, Tuple, Union
from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement
from numpy.random import RandomState

GRID_SIZE = 10


class GeneratedConfigSimple(FlatWorldConfig):
    '''
    Can generate random configurations, but it tries to put
    the agent close to a food.
    '''
    def __init__(self, elements, grid=None, seed=None):
        super().__init__()
        if grid is None:
            grid = [[None] * GRID_SIZE for r in range(GRID_SIZE)]
            for e in elements:
                gx, gy = e.position
                grid[gy][gx] = e
        self.elements = elements
        self.grid = grid
        self.seed(seed)

    @staticmethod
    def generate(num_food_elements: int, use_blue_food: bool, seed=None):
        elements = []
        grid = [[None] * GRID_SIZE for r in range(GRID_SIZE)]
        rs = RandomState()
        rs.seed(seed)
        GeneratedConfigSimple.populate_walls(elements, grid, rs)
        GeneratedConfigSimple.populate_food(elements, grid, num_food_elements, use_blue_food, rs)
        return GeneratedConfigSimple(elements, grid, rs.randint(0, 100000))

    @staticmethod
    def populate_food(elements, grid, num_food_elements: int, use_blue_food: bool, rs: RandomState):
        for i in range(num_food_elements + 1):
            is_final = i == num_food_elements
            for attempt in range(100):
                x, y = tuple(rs.randint(0, GRID_SIZE, size=2))
                if grid[y][x] is None:
                    final_food_fn = FlatWorldElement.final_food_2 if use_blue_food else FlatWorldElement.final_food
                    e = final_food_fn((x, y)) if is_final else FlatWorldElement.food((x, y))
                    elements.append(e)
                    grid[y][x] = e
                    break

    @staticmethod
    def populate_walls(elements: list, grid: List[list], rs: RandomState):
        num_walls = rs.randint(1, 4)
        for w in range(num_walls):
            diagonal = rs.uniform() < 0.2
            vertical = False if diagonal else rs.choice([True, False])
            length_limit = 0
            x1, y1 = 0, 0
            for attempt in range(10):
                x1, y1 = tuple(rs.randint(0, GRID_SIZE, size=2))
                length_limit = GRID_SIZE - (max(x1, y1) if diagonal else y1 if vertical else x1)
                if length_limit > 1:
                    break
            if length_limit <= 1:
                continue
            length = rs.randint(1, length_limit)
            we_func = GeneratedConfigSimple.random_wall_element_func(rs)
            for i in range(length):
                bx, by = x1, y1
                if diagonal:
                    bx, by = bx + i, by + i
                elif vertical:
                    by += i
                else:
                    bx += i
                if grid[by][bx] is None:
                    e = we_func((bx, by))
                    elements.append(e)
                    grid[by][bx] = e

    @staticmethod
    def random_wall_element_func(rs: RandomState):
        choice = rs.randint(0, 4)
        return [FlatWorldElement.fence_brick,
                FlatWorldElement.translucent_brick,
                FlatWorldElement.box,
                FlatWorldElement.wall_brick][choice]

    def get_initial_agent_position(self) -> Tuple[int, int]:
        g_w, g_h = self.get_grid_size()
        grid = self.grid
        fallback_x, fallback_y = 0, 0,
        for attempt in range(100):
            gx = self.random.randint(0, g_w)
            gy = self.random.randint(0, g_h)
            is_good = False
            if grid[gy][gx] is None:
                fallback_x, fallback_y = gx, gy
                for dx in [-1, 0, 1]:
                    check_gx = gx + dx
                    if 0 <= check_gx < g_w:
                        for dy in [-1, 0, 1]:
                            check_gy = gy + dy
                            if 0 <= check_gy < g_h:
                                e: FlatWorldElement = grid[check_gy][check_gx]
                                if e is not None and e.is_consumed:
                                    is_good = True
                                    break
                    if is_good:
                        break
            if is_good:
                return gx, gy,
        return fallback_x, fallback_y,

    def get_initial_agent_angle(self) -> float:
        return self.random.uniform() * 2 * math.pi

    def get_elements(self, agent_x, agent_y) -> List[object]:
        return [copy.deepcopy(e) for e in self.elements]

    def get_grid_size(self) -> Tuple[int, int]:
        return GRID_SIZE, GRID_SIZE,
