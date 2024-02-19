import unittest


class TestBoosterActuator(unittest.TestCase):
    def test_max_velocity(self, increment=0.05, friction_factor=0.84, num_boosters=2):
        velocity = 0
        for i in range(100):
            velocity += increment * num_boosters
            velocity *= friction_factor
        print(f'Final velocity: {velocity}')

    def test_max_angular_velocity(self, increment=0.1, friction_factor=0.84, num_boosters=2):
        velocity = 0
        for i in range(100):
            velocity += increment * num_boosters
            velocity *= friction_factor
        print(f'Final angular velocity: {velocity}')
