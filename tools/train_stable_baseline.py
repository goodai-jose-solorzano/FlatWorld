import numpy as np
from stable_baselines3 import PPO, DDPG, DQN, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv

from flat_world.ActuatorType import ActuatorType
from flat_world.FlatWorldEnvironment import FlatWorldEnvironment
from flat_world.tasks.simple_food_2_config import SimpleFood2Config


def evaluate(env, model: BaseAlgorithm, num_episodes: int, max_steps_per_episode: int):
    assert env.num_envs == 1, 'Assuming one env'
    total_reward = 0
    total_counts = 0
    for e in range(num_episodes):
        observation = env.reset()
        all_done = False
        reward_list = []
        states = None
        count = 0
        for s in range(max_steps_per_episode):
            if all_done:
                break
            action, states = model.predict(observation, states)
            observation, reward, done, info = env.step(action)
            reward_list.append(reward)
            all_done = np.all(done)
            count += 1
        reward_sum = np.sum(reward_list)
        total_reward += reward_sum
        total_counts += count
    mean_reward = total_reward / num_episodes
    print(f'Steps: {total_counts}')
    print(f'Evaluation reward/episode: {mean_reward:.4f}')


def run(num_eval_episodes=100, timesteps=100000):
    env_args = dict(
        config=SimpleFood2Config(),
        actuator_type=ActuatorType.DUAL,
        default_reward=-0.007,
    )
    env = make_vec_env(FlatWorldEnvironment, env_kwargs=env_args)
    print(f'Action space: {env.action_space}')
    model = PPO("MlpPolicy", env, verbose=0)
    print('Training...')
    model.learn(total_timesteps=timesteps)
    print('Evaluating...')
    evaluate(env, model, num_eval_episodes, 5000)


if __name__ == '__main__':
    run()
