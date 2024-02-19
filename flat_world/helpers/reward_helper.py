import numpy as np


def get_future_reward_matrix(reward_matrix: np.ndarray, includes_immediate=True):
    # Given m steps and n agents:
    # reward_matrix shape: (m, n,)

    future_rewards = \
        np.flip(
            np.cumsum(
                np.flip(reward_matrix, axis=0),
                axis=0),
            axis=0)
    if includes_immediate:
        return future_rewards
    future_rewards = future_rewards[1:, :]
    num_agents = reward_matrix.shape[1]
    return np.concatenate([future_rewards, np.zeros((1, num_agents))])


def get_discounted_reward_matrix(reward_matrix: np.ndarray, gamma: float):
    # Implements an optimized way of calculating the discounted reward.

    # Given m steps and n agents:
    # reward_matrix shape: (m, n,)
    num_steps, num_agents = reward_matrix.shape
    prev_dr = np.zeros((num_agents,))
    dr_matrix = np.empty((num_steps, num_agents,))
    for step_index in reversed(range(num_steps)):
        step_dr = reward_matrix[step_index] + prev_dr * gamma
        dr_matrix[step_index] = step_dr
        prev_dr = step_dr
    return dr_matrix


def get_lst_reward_matrix(reward_matrix: np.ndarray, r_weight: float):
    frm = get_future_reward_matrix(reward_matrix, includes_immediate=False)
    return reward_matrix * r_weight + frm * (1 - r_weight)
