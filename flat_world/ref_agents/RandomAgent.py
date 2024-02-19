from typing import Union
from typing import List

from flat_world.FlatWorldEnvironment import FlatWorldEnvironment
from flat_world.ref_agents.AbstractAgent import AbstractAgent


class RandomAgent(AbstractAgent):
    def __init__(self):
        pass

    def get_action(self, env: FlatWorldEnvironment, observation) -> Union[List[int], int]:
        if env.is_batched():
            return [env.action_space.sample() for _ in range(env.num_agents)]
        else:
            return env.action_space.sample()
