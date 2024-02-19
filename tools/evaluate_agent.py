from flat_world.FlatWorldEnvironment import FlatWorldEnvironment
from flat_world.tasks.starter_1_config import Starter1Config
from flat_world.tasks.starter_2_config import Starter2Config
from flat_world.helpers.ObservationScaler import ObservationScaler
from flat_world.ref_agents.AbstractAgent import AbstractAgent
from timeit import default_timer as timer

from flat_world.ref_agents.SimpleCNNAgent import SimpleCNNAgent


def evaluate(env, agent: AbstractAgent, num_episodes=100):
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


if __name__ == '__main__':
    env = FlatWorldEnvironment(Starter2Config())
    scaler = ObservationScaler()
    scaler.fit(env, 2000)
    agent = SimpleCNNAgent(scaler)
    env = FlatWorldEnvironment(Starter1Config())
    evaluate(env, agent)
