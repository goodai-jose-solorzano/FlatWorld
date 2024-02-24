from config_examples.eval.ChoiceTask import ChoiceTask
from flat_world.ActuatorType import ActuatorType
from flat_world.FlatWorldEnvironment import FlatWorldEnvironment
from flat_world.tasks.challenge_2_config import Challenge2Config

_params = dict(
    actuator_type=ActuatorType.DUAL,
    reward_no_action=0,
    default_reward=-0.004,
)
env = FlatWorldEnvironment(ChoiceTask(), **_params)

observation = env.reset()
while not env.is_closed:
    env.render()
    action = env.pull_next_keyboard_action()
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
