import math
from typing import List, Tuple
from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement
from tasks.EnvElement import EnvElement

_DEFAULT_ELEMENT_TYPES = [
    EnvElement.BOX,
    EnvElement.FENCE,
    EnvElement.YELLOW_FOOD,
    EnvElement.WALL,
    EnvElement.GREEN_FOOD,
    EnvElement.BLUE_FOOD,
]


class BoxPhysicsTask(FlatWorldConfig):
    def __init__(self, grid_size: int, num_boxes: int, num_elements: int,
                 element_types=None):
        super().__init__()
        if grid_size < 9:
            raise Exception('Grid too small!')
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_elements = num_elements
        if element_types is None:
            element_types = _DEFAULT_ELEMENT_TYPES
        self.element_types = element_types

    def get_grid_size(self) -> Tuple[int, int]:
        return self.grid_size, self.grid_size,

    def get_initial_agent_position(self) -> Tuple[int, int]:
        return self.random.randint(0, self.grid_size), self.random.randint(self.grid_size - 4, self.grid_size),

    def get_initial_agent_angle(self) -> float:
        return self.random.uniform(-math.pi / 3, -math.pi * 2 / 3)

    def get_elements(self, agent_x: int, agent_y: int) -> List[FlatWorldElement]:
        elements = []
        box_row = self.random.randint(agent_y - 3, agent_y)
        box_x = self.random.choice(range(0, self.grid_size), size=self.num_boxes, replace=False)
        for x in box_x:
            elements.append(FlatWorldElement.box((x, box_row)))
        positions = [(x, y) for x in range(0, self.grid_size) for y in range(box_row - 2, box_row)]
        self.random.shuffle(positions)
        for i in range(self.num_elements):
            element_type: EnvElement = self.random.choice(self.element_types)
            cfn = element_type.get_constructor_fn()
            element = cfn(tuple(positions[i]))
            elements.append(element)
        return elements

