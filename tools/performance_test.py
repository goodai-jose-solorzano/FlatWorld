from flat_world.FlatWorldEnvironment import FlatWorldEnvironment
from flat_world.tasks.generic_config import GenericConfig
from timeit import default_timer as timer
from flat_world.ref_agents.RandomAgent import RandomAgent


def run_trials(time_limit: float):
    agent = RandomAgent()
    step_count = 0
    episode_count = 0
    env = FlatWorldEnvironment(GenericConfig.generate())
    observation = env.reset()
    agent.env_was_reset(env, observation)
    time_start = timer()
    while True:
        action = agent.get_action(env, observation)
        observation, reward, done, info = env.step(action)
        step_count += 1
        if done:
            episode_count += 1
            time_end = timer()
            elapsed = time_end - time_start
            if episode_count % 20 == 0:
                env.close()
                env = FlatWorldEnvironment(GenericConfig.generate())
            observation = env.reset()
            agent.env_was_reset(env, observation)
            if elapsed >= time_limit:
                return step_count, episode_count, elapsed,


if __name__ == '__main__':
    # Warmup steps
    run_trials(3)
    # Timed steps
    sc, ec, et = run_trials(20)
    print('Speed: %.1f steps/sec | %.1f episodes/sec' % (sc / et, ec / et))
    print('Steps per episode: %.1f' % (sc / ec))
