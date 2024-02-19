from typing import List

import numpy as np
import torch
from torch import optim
from flat_world.FlatWorldEnvironment import FlatWorldEnvironment
from flat_world.tasks.multi_agent.food_for_all_config import FoodForAllConfig
from flat_world.tasks.multi_agent.multiagent_all_config import MultiAgentAllConfig
from flat_world.tasks.starter_1_config import Starter1Config
from flat_world.tasks.starter_2_config import Starter2Config
from flat_world.helpers.ObservationScaler import ObservationScaler
from flat_world.ref_agents.AbstractAgent import AbstractAgent
from flat_world.ref_agents.SimpleCNNAgent import SimpleCNN1D, SimpleCNNAgent, SimpleCNNActionProbAgent
from timeit import default_timer as timer

# Multi-Agent
# Optimization by Vanilla Policy Gradient


def evaluate(env: FlatWorldEnvironment, agent: AbstractAgent, num_episodes=50):
    num_agents = env.num_agents
    reward_sum = np.zeros((num_agents,), dtype=np.float16)
    time1 = timer()
    for e in range(num_episodes):
        observation = env.reset()
        done = np.full((num_agents,), False, dtype=bool)
        total_reward = np.zeros((num_agents,), dtype=np.float16)
        while not np.all(done):
            action = agent.get_action(env, observation)
            prev_done = done
            observation, reward, done, _ = env.step(action)
            total_reward += (reward * ~prev_done)
        reward_sum += total_reward
    time2 = timer()
    speed = num_episodes / (time2 - time1)
    episode_reward = np.mean(reward_sum) / num_episodes
    print('Speed: %.1f episodes / second' % speed)
    print('Score: %.2f rewards / episode / agent' % episode_reward)


def rewards_to_quality_tensor(reward_matrix: np.ndarray,
                              r_weight: float = 0.50) -> torch.Tensor:
    # Given n agents and m steps:
    #   reward_matrix shape: (m, n,)
    # Returns tensor of shape (m, n,)

    future_rewards = \
        np.flip(
            np.cumsum(
                np.flip(reward_matrix, axis=0),
                axis=0),
            axis=0)
    q = reward_matrix * r_weight + future_rewards * (1 - r_weight)
    return torch.from_numpy(q).float()


def done_to_validity_tensor(done_matrix: np.ndarray):
    batch_size = done_matrix.shape[1]
    # First 'done=True' counts toward observed quality/value.
    first_done_idx = np.argmax(done_matrix, 0)
    # first_done_idx are step indexes
    done_copy = np.copy(done_matrix)
    done_copy[first_done_idx, np.arange(0, batch_size)] = False
    return torch.from_numpy(~done_copy).bool()


def train(env, scaler, num_episodes=600, lr=0.001):
    num_agents = env.num_agents
    network = SimpleCNN1D()
    optimizer = optim.Adam(network.parameters(), lr=lr)
    loss_ra = None
    reward_ra = None
    for e in range(num_episodes):
        observation = env.reset()
        action_prob_list = []
        reward_list = []
        done_list = []
        episode_reward = np.zeros((num_agents,))
        done = np.full((num_agents,), False, dtype=bool)
        network.train()
        network.zero_grad()
        while not np.all(done):
            std_obs = scaler.transform(observation)
            input_tensor = torch.from_numpy(std_obs).float()
            action_logits = network(input_tensor)
            batch_size = action_logits.size()[0]
            # action_logits: (B, 5,)
            action_probs = torch.softmax(action_logits, 1)
            actions = torch.multinomial(action_probs, 1).squeeze(1)
            # actions shape: (B,)
            selected_action_probs = action_probs[torch.arange(0, batch_size), actions]
            # selected_action_probs shape: (B,)
            action_prob_list.append(selected_action_probs)
            prev_done = done
            observation, reward, done, _ = env.step(actions.numpy())
            # reward/done shape: (B,)
            reward_list.append(reward)
            done_list.append(done)
            episode_reward += reward * ~prev_done
        act_prob_t = torch.stack(action_prob_list)
        # act_prob_log_t shape: (S, B,)
        reward_matrix = np.array(reward_list)
        done_matrix = np.array(done_list)
        obs_q_t = rewards_to_quality_tensor(reward_matrix)
        good_t = done_to_validity_tensor(done_matrix)
        # reward_matrix/done_matrix/osq_s_t shape: (S, B,)
        loss = -torch.sum(obs_q_t * torch.log(act_prob_t) * good_t) / torch.sum(good_t)
        loss.backward()
        optimizer.step()
        loss_scalar = loss.item()
        if loss_ra is None:
            loss_ra = loss_scalar
        else:
            loss_ra = loss_ra * 0.9 + loss_scalar * 0.1
        agent_reward = np.mean(episode_reward)
        if reward_ra is None:
            reward_ra = agent_reward
        else:
            reward_ra = reward_ra * 0.9 + agent_reward * 0.1
        if e % 20 == 0:
            print(f'Episode {e} loss RA: {loss_ra} | reward RA: {reward_ra}')
    return network


if __name__ == '__main__':
    scaler_env = FlatWorldEnvironment(MultiAgentAllConfig(), num_agents=10)
    print('Training scaler...')
    scaler = ObservationScaler()
    scaler.fit(scaler_env, 1000)
    print('Training agent...')
    env = FlatWorldEnvironment(FoodForAllConfig(), num_agents=10)
    network = train(env, scaler)
    print('Evaluating...')
    agent = SimpleCNNActionProbAgent(scaler, network)
    evaluate(env, agent)
