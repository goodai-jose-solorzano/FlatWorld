import math
import unittest

import numpy as np

from flat_world.DualActuator import DualActuator
from flat_world.helpers.PrecalculatedOrientations import PrecalculatedOrientations


class TestDualActuator(unittest.TestCase):
    def test_distance(self):
        num_steps = 56
        oh = PrecalculatedOrientations(3, 20, math.pi / 3)
        actuator = DualActuator(oh, math.pi / 4)
        position = np.array([0, 0])
        for step in range(num_steps):
            action = [1, 0]
            new_position, has_movement, no_action = actuator.apply_action(action, position)
            position = new_position
        distance = np.linalg.norm(position)
        # Expected: 13.4
        print(f'Distance: {distance}')

    def test_deceleration_full(self):
        num_steps = 15
        oh = PrecalculatedOrientations(3, 20, math.pi / 3)
        actuator = DualActuator(oh, math.pi / 4)
        position = np.array([0, 0])
        for step in range(num_steps):
            action = [1, 0]
            new_position, has_movement, no_action = actuator.apply_action(action, position)
            position = new_position
        count_stop = 0
        while True:
            action = [0, 0]
            new_position, has_movement, no_action = actuator.apply_action(action, position)
            diff = np.linalg.norm(new_position - position)
            if np.isclose(diff, 0):
                break
            position = new_position
            count_stop += 1
        # Expected: 8
        print(f'Count: {count_stop}')

    def test_deceleration_one(self):
        num_steps = 1
        oh = PrecalculatedOrientations(3, 20, math.pi / 3)
        actuator = DualActuator(oh, math.pi / 4)
        position = np.array([0, 0])
        for step in range(num_steps):
            action = [1, 0]
            new_position, has_movement, no_action = actuator.apply_action(action, position)
            position = new_position
        count_stop = 0
        while True:
            action = [0, 0]
            new_position, has_movement, no_action = actuator.apply_action(action, position)
            diff = np.linalg.norm(new_position - position)
            if np.isclose(diff, 0):
                break
            position = new_position
            count_stop += 1
        # Expected: 2
        print(f'Count: {count_stop}')
