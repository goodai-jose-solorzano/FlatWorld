from typing import List, Union

import numpy as np
import torch

from flat_world.FlatWorldEnvironment import FlatWorldEnvironment
from flat_world.helpers.ObservationScaler import ObservationScaler
from flat_world.ref_agents.AbstractAgent import AbstractAgent
from torch import nn

ACTIONS = list(range(5))


class SimpleCNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 8, (5,), padding=2)
        self.pool1 = nn.MaxPool1d((2,))
        self.dense1 = nn.Linear(80, 100)
        self.dense2 = nn.Linear(100, 5)

    def forward(self, std_observation):
        # std_observation: 1x4x20
        x = self.conv1(std_observation)
        # x: 1x8x20
        x = self.pool1(x)
        # x : 1x8x10
        x = torch.flatten(x, 1)
        # x : 1x80
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        # Returns action logits (1x5)
        return x


class SimpleCNNAgent(AbstractAgent):
    def __init__(self, scaler: ObservationScaler, network: SimpleCNN1D = None):
        self.scaler = scaler
        self.network = SimpleCNN1D() if network is None else network

    def get_action(self, env: FlatWorldEnvironment, observation) -> int:
        is_batched = env.is_batched()
        std_obs = self.scaler.transform(observation)
        input_tensor = torch.from_numpy(std_obs).float()
        if not is_batched:
            input_tensor = input_tensor.unsqueeze(0)
        action_logits = self.network(input_tensor)
        # action_logits: 1x5
        actions = torch.argmax(action_logits)
        actions = actions.detach().cpu().numpy()
        if not is_batched:
            actions = actions.squeeze(0).item()
        return actions


class SimpleCNNActionProbAgent(AbstractAgent):
    def __init__(self, scaler: ObservationScaler, network: SimpleCNN1D = None):
        self.scaler = scaler
        self.network = SimpleCNN1D() if network is None else network

    def get_action(self, env, observation) -> Union[List[int], int]:
        is_batched = env.is_batched()
        std_obs = self.scaler.transform(observation)
        input_tensor = torch.from_numpy(std_obs).float()
        if not is_batched:
            input_tensor = input_tensor.unsqueeze(0)
        action_logits = self.network(input_tensor)
        action_probs = torch.softmax(action_logits, 1)
        action_probs_np = action_probs.detach().cpu().numpy()
        if not is_batched:
            action_probs_np = action_probs_np.squeeze(0)
            return np.random.choice(ACTIONS, p=action_probs_np)
        else:
            num_agents = env.num_agents
            actions = [None] * num_agents
            for i in range(num_agents):
                actions[i] = np.random.choice(ACTIONS, p=action_probs_np[i])
            return actions
