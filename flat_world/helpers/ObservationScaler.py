import numpy as np

from flat_world.FlatWorldEnvironment import FlatWorldEnvironment


class ObservationScaler:
    def __init__(self):
        self.means: np.ndarray = None
        self.stdevs: np.ndarray = None

    def fit(self, env: FlatWorldEnvironment, num_steps: int):
        is_batched = env.is_batched()
        num_agents = env.num_agents
        observation = env.reset()
        obs_shape = observation.shape
        resolution = obs_shape[2 if is_batched else 1]
        obs_list = []
        if is_batched:
            obs_list.extend(list(observation))
        else:
            obs_list.append(observation)
        for s in range(num_steps):
            actions = [env.action_space.sample() for _ in range(num_agents)]
            if not is_batched:
                actions = actions[0]
            observation, _, done, info = env.step(actions)
            should_reset = np.all(done) if is_batched else done
            if should_reset:
                observation = env.reset()
            if is_batched:
                valid_observations = observation[~done]
                obs_list.extend(list(valid_observations))
            else:
                obs_list.append(observation)
        base_means = np.mean(np.mean(obs_list, 2), 0)
        base_variances = np.mean(np.var(obs_list, 2), 0)
        base_stdevs = np.sqrt(base_variances)
        self.means = np.tile(base_means, (resolution, 1,)).transpose()
        self.stdevs = np.tile(base_stdevs, (resolution, 1,)).transpose()

    def transform(self, observation):
        return (observation - self.means) / self.stdevs
