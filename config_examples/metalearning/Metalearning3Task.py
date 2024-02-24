import math
from typing import List, Tuple, Union

import numpy as np
from flat_world.FlatWorldElement import FlatWorldElement

from tasks.FinalFoodType import FinalFoodType
from tasks.metalearning.BaseMetalearningTask import BaseMetalearningTask


class Metalearning3Task(BaseMetalearningTask):
    def __init__(self, grid_size: Tuple[int, int], num_food_elements: int, final_food_type: FinalFoodType):
        super().__init__()
        self.grid_size = grid_size
        self.num_food_elements = num_food_elements
        self.final_food_type = final_food_type
        gw, gh = grid_size
        self.all_positions = [(x, y) for x in range(gw) for y in range(gh)]

    def get_grid_size(self) -> Tuple[int, int]:
        return self.grid_size

    def get_reward_upper_bound(self):
        r = 0.5 if self.final_food_type == FinalFoodType.BLUE else 1.0
        r += self.num_food_elements * 1.0
        return r

    def get_initial_agent_angles(self, num_agents: int) -> Union[List[float], np.ndarray]:
        return self.random.uniform(0, math.pi * 2, size=num_agents)

    def get_initial_agent_positions(self, num_agents: int) -> List[Tuple[int, int]]:
        return Metalearning3Task.pos_choice(self.random, self.all_positions, num_agents)

    def get_elements_for_agents(self, initial_agent_positions: List[Tuple[int, int]]) -> List[FlatWorldElement]:
        nearby_pool = []
        gw, gh = self.get_grid_size()
        for a_pos in initial_agent_positions:
            ax, ay = a_pos
            for x in range(ax - 2, ax + 3):
                if 0 <= x < gw:
                    for y in range(ay - 2, ay + 3):
                        if 0 <= y < gh:
                            nearby_pool.append((x, y,))
        nearby_set = set(nearby_pool)
        nearby_set = nearby_set.difference(set(initial_agent_positions))
        elements = []
        food_pos = Metalearning3Task.pos_choice(self.random, list(nearby_set), self.num_food_elements)
        for pos in food_pos:
            elements.append(FlatWorldElement.food(pos))
        fft = self.final_food_type
        num_ff = len(initial_agent_positions) * fft.get_num_elements()
        remaining_pos_set = nearby_set.difference(set(food_pos))
        ff_pos = Metalearning3Task.pos_choice(self.random, list(remaining_pos_set), num_ff)
        ff_elem_fn = fft.get_element_fns()
        for pos_idx, pos in enumerate(ff_pos):
            fn = ff_elem_fn[pos_idx % len(ff_elem_fn)]
            elements.append(fn(pos))
        return elements
