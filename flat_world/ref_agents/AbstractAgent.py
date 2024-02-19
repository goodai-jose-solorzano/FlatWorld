from abc import ABC, abstractmethod
from typing import List, Union


class AbstractAgent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, env, observation) -> Union[List[int], int]:
        pass

    def env_was_reset(self, env, observation):
        pass
