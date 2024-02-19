import numpy as np
import torch
from torch import optim

from flat_world.ActuatorType import ActuatorType
from flat_world.FlatWorldEnvironment import FlatWorldEnvironment
from flat_world.helpers.reward_helper import get_discounted_reward_matrix
from flat_world.ref_agents.DualCNN import DualCNN
from flat_world.tasks.all_env_elements_config import AllEnvElementsConfig
from flat_world.tasks.simple_food_2_config import SimpleFood2Config
from flat_world.tasks.simple_food_config import SimpleFoodConfig
from flat_world.tasks.starter_1_config import Starter1Config
from flat_world.tasks.starter_2_config import Starter2Config
from flat_world.helpers.ObservationScaler import ObservationScaler
from flat_world.ref_agents.AbstractAgent import AbstractAgent
from flat_world.ref_agents.SimpleCNNAgent import SimpleCNN1D, SimpleCNNAgent, SimpleCNNActionProbAgent
from timeit import default_timer as timer

# Single-Agent
# Optimization by Vanilla Policy Gradient
# Dual actuator


def rewards_to_value_tensor(rewards: list, gamma=0.993):
    dr_matrix = get_discounted_reward_matrix(np.array(rewards)[:, None], gamma)
    return torch.from_numpy(dr_matrix).float().squeeze(1)


def actor_loss(action_prob: torch.Tensor, advantage: torch.Tensor, clip=0.08):
    # Clipped loss (PPO)
    action_prob = torch.clamp(action_prob, min=1e-10)
    prob_ratio = action_prob / action_prob.detach()
    bounded_ratio = torch.clamp(prob_ratio, min=1.0 - clip, max=1.0 + clip)
    min_ratio: torch.Tensor = torch.min(prob_ratio, bounded_ratio)
    return -torch.mean(min_ratio * advantage)


def obs_tensor(std_obs: np.ndarray):
    std_obs_t = torch.from_numpy(std_obs).float()
    return std_obs_t.unsqueeze(0)


def evaluate(env, scaler, network: DualCNN, num_episodes: int):
    network.eval()
    actions = list(range(3))
    total_reward = 0
    for e in range(num_episodes):
        observation = env.reset()
        done = False
        reward_list = []
        while not done:
            std_obs = scaler.transform(observation)
            input_tensor = obs_tensor(std_obs)
            motion_output, rot_output = network(input_tensor)
            motion_action_logits = motion_output.squeeze(0)
            rot_action_logits = rot_output.squeeze(0)
            motion_action_probs = torch.softmax(motion_action_logits, 0)
            rot_action_probs = torch.softmax(rot_action_logits, 0)
            motion_action_probs_np = motion_action_probs.detach().cpu().numpy()
            rot_action_probs_np = rot_action_probs.detach().cpu().numpy()
            motion_action = np.random.choice(actions, p=motion_action_probs_np)
            rot_action = np.random.choice(actions, p=rot_action_probs_np)
            action = [motion_action, rot_action]
            observation, reward, done, _ = env.step(action)
            reward_list.append(reward)
        reward_sum = np.sum(reward_list)
        total_reward += reward_sum
    mean_reward = total_reward / num_episodes
    print(f'Evaluation reward: {mean_reward:.4f}')


def train(env, scaler, num_episodes: int, lr=0.0001, motion_weight=0.5):
    network = DualCNN()
    optimizer = optim.AdamW(network.parameters(), lr=lr, weight_decay=0.0003)
    # optimizer = optim.Adam(network.parameters(), lr=lr)
    loss_ra = None
    reward_ra = None
    actions = list(range(3))
    for e in range(num_episodes):
        observation = env.reset()
        done = False
        motion_action_prob_list = []
        rot_action_prob_list = []
        reward_list = []
        network.train()
        network.zero_grad()
        while not done:
            std_obs = scaler.transform(observation)
            input_tensor = obs_tensor(std_obs)
            motion_output, rot_output = network(input_tensor)
            motion_action_logits = motion_output.squeeze(0)
            rot_action_logits = rot_output.squeeze(0)
            motion_action_probs = torch.softmax(motion_action_logits, 0)
            rot_action_probs = torch.softmax(rot_action_logits, 0)
            motion_action_probs_np = motion_action_probs.detach().cpu().numpy()
            rot_action_probs_np = rot_action_probs.detach().cpu().numpy()
            motion_action = np.random.choice(actions, p=motion_action_probs_np)
            rot_action = np.random.choice(actions, p=rot_action_probs_np)
            motion_action_prob_list.append(motion_action_probs[motion_action])
            rot_action_prob_list.append(rot_action_probs[rot_action])
            action = [motion_action, rot_action]
            observation, reward, done, _ = env.step(action)
            reward_list.append(reward)
        value_t = rewards_to_value_tensor(reward_list)
        motion_ap_t = torch.stack(motion_action_prob_list)
        rot_ap_t = torch.stack(rot_action_prob_list)
        motion_loss = actor_loss(motion_ap_t, value_t)
        rot_loss = actor_loss(rot_ap_t, value_t)
        loss = motion_loss * motion_weight + rot_loss * (1 - motion_weight)
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
        if e % 5 == 0:
            print(f'Episode {e} loss RA: {loss_ra} | reward RA: {reward_ra}')
    return network


if __name__ == '__main__':
    scaler_env = FlatWorldEnvironment(Starter2Config())
    print('Training scaler...')
    scaler = ObservationScaler()
    scaler.fit(scaler_env, 2000)
    print('Training agent...')
    env = FlatWorldEnvironment(SimpleFood2Config(), actuator_type=ActuatorType.DUAL, default_reward=-0.007)
    num_episodes = 2000
    network = train(env, scaler, num_episodes)
    print('Evaluating...')
    num_eval_episodes = 200
    evaluate(env, scaler, network, num_eval_episodes)
