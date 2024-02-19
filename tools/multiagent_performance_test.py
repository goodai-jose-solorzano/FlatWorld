import numpy as np

from flat_world.FlatWorldEnvironment import FlatWorldEnvironment
from timeit import default_timer as timer
from flat_world.ref_agents.RandomAgent import RandomAgent
from flat_world.tasks.multi_agent.big_grid_config import BigGridConfig


def run_trials(time_limit: float):
    agent = RandomAgent()
    step_count = 0
    episode_count = 0
    num_agents = 50
    env = FlatWorldEnvironment(BigGridConfig(), num_agents=num_agents, obs_resolution=20)
    observation = env.reset()
    agent.env_was_reset(env, observation)
    obs_count = 0
    time_start = timer()
    while True:
        action = agent.get_action(env, observation)
        observation, reward, done, info = env.step(action)
        step_count += 1
        obs_count += num_agents
        if np.all(done):
            episode_count += 1
            observation = env.reset()
            agent.env_was_reset(env, observation)
        time_end = timer()
        elapsed = time_end - time_start
        if elapsed >= time_limit:
            return step_count, obs_count, episode_count, elapsed,


if __name__ == '__main__':
    # Warmup steps
    run_trials(3)
    # Timed steps
    sc, oc, ec, et = run_trials(20)
    print('Speed: %.1f steps/sec | %.1f episodes/sec | %.1f observations.sec' % (sc / et, ec / et, oc / et))
    if ec == 0:
        print('No episodes completed!')
    else:
        print('Steps per episode: %.1f' % (sc / ec))
