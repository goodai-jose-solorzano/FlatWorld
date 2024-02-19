from abc import abstractmethod, ABC
from typing import List

import numpy as np
import torch


class AbstractAutonomousAgent(ABC):
    # An agent that can act and learn by calling step() on it.
    # It keeps track of episode rewards, done states and action information.

    def __init__(self):
        self.env = None
        self.learning_enabled = True
        self.observation = None
        self.action = None
        self.reward = None
        self.done = None
        self.info = None
        self.reward_list: List[np.ndarray] = []
        self.done_list = []
        self.info_list = []
        self.action_info_list = []
        self.reset_all()

    def reset_all(self):
        self.reset_environment()
        self.agent_was_reset()

    def set_environment(self, env):
        if self.env is not None:
            self.env.close()
        self.env = env
        self.reset_environment()

    def set_learning_enabled(self, value: bool):
        self.learning_enabled = value
        if not value:
            self.learning_was_disabled()
            self.clear_learning_queues()

    def reset_environment(self, inform_done=True):
        if inform_done:
            self._inform_episode_done(False)
        self.clear_learning_queues()
        self.observation = self.env.reset() if self.env is not None else None
        self.environment_was_reset()

    def clear_learning_queues(self):
        self.reward_list.clear()
        self.done_list.clear()
        self.info_list.clear()
        self.action_info_list.clear()

    def done_to_validity_matrix(self, done_matrix: np.ndarray):
        batch_size = done_matrix.shape[1]
        # First 'done=True' counts toward observed quality/value.
        first_done_idx = np.argmax(done_matrix, 0)
        # first_done_idx are step indexes
        done_copy = np.copy(done_matrix)
        done_copy[first_done_idx, np.arange(0, batch_size)] = False
        return ~done_copy

    def done_to_validity_tensor(self, done_matrix: np.ndarray):
        return torch.from_numpy(self.done_to_validity_matrix(done_matrix)).bool()

    def step(self, render=False) -> bool:
        '''
        Automatically executes an environment step.
        If all agents are done, episode_done() is called.
        :return: True if all agents are done.
        '''
        if render:
            self.env.render()
        action_info = self.get_action_info()
        self.action = action_info['actions']
        self.observation, self.reward, self.done, self.info = self.env.step(self.action)
        self.reward_list.append(self.reward)
        self.done_list.append(self.done)
        self.info_list.append(self.info)
        self.action_info_list.append(action_info)
        all_done = np.all(self.done)
        if all_done:
            self._inform_episode_done(natural=True)
            self.reset_environment(inform_done=False)
        return all_done

    def _inform_episode_done(self, natural: bool):
        self.episode_rewards = np.sum(self.reward_list)
        self.episode_done(natural)

    def get_last_episode_rewards(self):
        return self.episode_rewards

    @abstractmethod
    def environment_was_reset(self):
        pass

    @abstractmethod
    def agent_was_reset(self):
        pass

    @abstractmethod
    def episode_done(self, natural=True):
        pass

    @abstractmethod
    def learning_was_disabled(self):
        pass

    @abstractmethod
    def get_action_info(self) -> dict:
        '''
        :return: A dictionary with an actions property.
        '''
        pass
