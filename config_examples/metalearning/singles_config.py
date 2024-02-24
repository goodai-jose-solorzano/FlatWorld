import copy
import math
import numpy as np
from typing import List, Tuple, Union
from flat_world.FlatWorldConfig import FlatWorldConfig

GRID_SIZE = 10


class SinglesConfig(FlatWorldConfig):
    def __init__(self, create_functions):
        super().__init__()
        self.create_functions = create_functions

    @staticmethod
    def generate(element_creation_functions: list):
        return SinglesConfig(element_creation_functions)

    def get_grid_size(self) -> Tuple[int, int]:
        return GRID_SIZE, GRID_SIZE,

    def get_initial_agent_position(self) -> Tuple[int, int]:
        gw, gh = self.get_grid_size()
        return self.random.randint(0, gw), self.random.randint(0, gh),

    def get_initial_agent_angle(self) -> float:
        return self.random.uniform() * 2 * math.pi

    def get_elements(self, agent_x, agent_y) -> List[object]:
        gw, gh = self.get_grid_size()
        num_positions = gw * gh
        a_position = agent_y * gw + agent_x
        all_positions = list(range(num_positions))
        del all_positions[a_position]
        num_elements = len(self.create_functions)
        sampled_positions = self.random.choice(all_positions, num_elements, replace=False)
        elements = []
        for i, fn in enumerate(self.create_functions):
            position = sampled_positions[i]
            x, y = position % gw, position // gw
            elements.append(fn((x, y)))
        return elements
