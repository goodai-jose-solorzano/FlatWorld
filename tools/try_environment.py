from flat_world.ActuatorType import ActuatorType
from flat_world.FlatWorldEnvironment import FlatWorldEnvironment
from flat_world.tasks.simple_food_2_config import SimpleFood2Config

_params = dict(
    actuator_type=ActuatorType.DUAL,
    reward_no_action=0,
    default_reward=-0.004,
)
env = FlatWorldEnvironment(SimpleFood2Config(), **_params)

observation = env.reset()
while not env.is_closed:
    env.render()
    action = env.pull_next_keyboard_action()
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
