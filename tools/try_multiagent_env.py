import math
import numpy as np

from flat_world.ActuatorType import ActuatorType
from flat_world.FlatWorldEnvironment import FlatWorldEnvironment
from flat_world.tasks.multi_agent.big_grid_config import BigGridConfig
from flat_world.tasks.multi_agent.food_for_all_config import FoodForAllConfig
from flat_world.tasks.multi_agent.multiagent_all_config import MultiAgentAllConfig

_at = ActuatorType.BOOSTERS_4
env = FlatWorldEnvironment(MultiAgentAllConfig(), actuator_type=_at,
                           reward_no_action=0, num_agents=3, obs_resolution=50)

observation = env.reset()
while not env.is_closed:
    env.render()
    action = env.pull_next_keyboard_action()
    total_reward_before = env.get_total_reward_of_selected_agent()
    observation, reward, done, info = env.step(action)
    reward_selected = reward[env.selected_agent_index]
    total_reward_after = env.get_total_reward_of_selected_agent()
    assert math.isclose(total_reward_after - total_reward_before, reward_selected, abs_tol=0.0001)
    if np.all(done):
        observation = env.reset()
