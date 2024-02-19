import numpy as np
import torch
from torch import optim
from flat_world.FlatWorldEnvironment import FlatWorldEnvironment
from flat_world.tasks.all_env_elements_config import AllEnvElementsConfig
from flat_world.tasks.simple_food_config import SimpleFoodConfig
from flat_world.tasks.starter_1_config import Starter1Config
from flat_world.tasks.starter_2_config import Starter2Config
from flat_world.helpers.ObservationScaler import ObservationScaler
from flat_world.ref_agents.AbstractAgent import AbstractAgent
from flat_world.ref_agents.SimpleCNNAgent import SimpleCNN1D, SimpleCNNAgent, SimpleCNNActionProbAgent
from timeit import default_timer as timer

# Single-Agent
# Optimization by Vanilla Policy Gradient


def evaluate(env, agent: AbstractAgent, num_episodes=300):
    reward_sum = 0
    time1 = timer()
    for e in range(num_episodes):
        observation = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.get_action(env, observation)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
        reward_sum += total_reward
    time2 = timer()
    speed = num_episodes / (time2 - time1)
    episode_reward = reward_sum / num_episodes
    print('Speed: %.1f episodes / second' % speed)
    print('Score: %.2f rewards / episode' % episode_reward)


def rewards_to_value_tensor(rewards: list, r_weight: float = 0.25):
    value = 0
    value_list = []
    for r in reversed(rewards):
        value += r
        value_list.insert(0, value)
    values_t = torch.Tensor(value_list)
    rewards_t = torch.Tensor(rewards)
    return rewards_t * r_weight + values_t * (1 - r_weight)


def train(env, scaler, num_episodes=2000, lr=0.0003):
    network = SimpleCNN1D()
    optimizer = optim.Adam(network.parameters(), lr=lr)
    loss_ra = None
    reward_ra = None
    actions = list(range(5))
    for e in range(num_episodes):
        observation = env.reset()
        done = False
        log_action_prob_list = []
        reward_list = []
        network.train()
        network.zero_grad()
        while not done:
            std_obs = scaler.transform(observation)
            input_tensor = torch.from_numpy(std_obs).unsqueeze(0).float()
            action_logits = network(input_tensor)
            action_logits = action_logits.squeeze(0)
            action_probs = torch.softmax(action_logits, 0)
            action_probs_np = action_probs.detach().cpu().numpy()
            action = np.random.choice(actions, p=action_probs_np)
            log_action_prob_list.append(torch.log(action_probs[action]))
            observation, reward, done, _ = env.step(action)
            reward_list.append(reward)
        value_t = rewards_to_value_tensor(reward_list)
        aplog_t = torch.stack(log_action_prob_list)
        loss = -torch.mean(value_t * aplog_t)
        loss.backward()
        optimizer.step()
        loss_scalar = loss.item()
        if loss_ra is None:
            loss_ra = loss_scalar
        else:
            loss_ra = loss_ra * 0.9 + loss_scalar * 0.1
        sum_rewards = np.sum(reward_list)
        if reward_ra is None:
            reward_ra = sum_rewards
        else:
            reward_ra = reward_ra * 0.9 + sum_rewards * 0.1
        if e % 50 == 0:
            print(f'Episode {e} loss RA: {loss_ra} | reward RA: {reward_ra}')
    return network


if __name__ == '__main__':
    scaler_env = FlatWorldEnvironment(Starter2Config())
    print('Training scaler...')
    scaler = ObservationScaler()
    scaler.fit(scaler_env, 2000)
    print('Training agent...')
    env = FlatWorldEnvironment(Starter1Config())
    network = train(env, scaler)
    print('Evaluating...')
    agent = SimpleCNNActionProbAgent(scaler, network)
    evaluate(env, agent)
