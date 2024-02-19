import io
import math
import os
import tkinter
from tkinter import BOTH, TOP, NW, messagebox

import numpy as np
from typing import Tuple, Union, List, Optional

from PIL import ImageDraw, ImageTk
from PIL import Image
from PIL import ImageFont
from gym import Env, spaces
from numpy.random import RandomState

from flat_world import FlatWorldConfig
from flat_world.ActuatorType import ActuatorType
from flat_world.AgentState import AgentState
from flat_world.BaseEnvCharacter import BaseEnvCharacter, ACTION_NONE, ACTION_FORWARD, ACTION_BACK, ACTION_LEFT, \
    ACTION_RIGHT, ACTION_TOGGLE
from flat_world.EnvState import EnvState
from flat_world.PredatorLevel import PredatorLevel
from flat_world.helpers.PrecalculatedOrientations import PrecalculatedOrientations


class FlatWorldEnvironment(Env):
    def __init__(self, config: FlatWorldConfig,
                 obs_resolution: int = 20, num_agents: int = 1,
                 default_reward=-0.01, reward_no_action=-0.005,
                 reward_toggle=-0.002, reward_predator=-1.0,
                 predator_level=PredatorLevel.MEDIUM,
                 actuator_type=ActuatorType.DEFAULT,
                 window_size: Tuple[int, int] = None,
                 squeeze=True):
        super().__init__()
        assert num_agents >= 1, 'num_agents must be greater than zero!'
        self.config = config
        self.random = RandomState()
        self.num_agents = num_agents
        self.obs_resolution = obs_resolution
        self.predator_level = predator_level
        self.actuator_type = actuator_type
        self.window_size = window_size
        self.metadata = {'render.modes': ['human', 'rgba_array', 'rgb_array']}
        self.action_space = FlatWorldEnvironment._action_space(actuator_type)
        if num_agents > 1 or not squeeze:
            self.observation_space = spaces.Box(low=np.zeros((num_agents, 4, obs_resolution), dtype=np.float32),
                                                high=np.full((num_agents, 4, obs_resolution), 255, dtype=np.float32),
                                                dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=np.zeros((4, obs_resolution), dtype=np.float32),
                                                high=np.full((4, obs_resolution), 255, dtype=np.float32),
                                                dtype=np.float32)
        self.default_reward = default_reward
        self.reward_no_action = reward_no_action
        self.reward_toggle = reward_toggle
        self.reward_predator = reward_predator
        self.squeeze = squeeze
        self.env_state: EnvState = None
        self.is_closed = False
        self.min_cumulative_reward: float = 0
        self.cumulative_reward_prev_episode = None
        self.selected_agent_index = 0
        self.bkg_color = None
        self.window: Optional[tkinter.Tk] = None
        self.canvas = None
        self.tk_agent_var = None
        self.cached_font = None
        self.keyboard_actions = []

    @staticmethod
    def _action_space(actuator_type: ActuatorType) -> spaces.Space:
        if actuator_type == ActuatorType.DEFAULT:
            return spaces.Discrete(6)
        elif actuator_type == ActuatorType.DUAL:
            return spaces.MultiDiscrete([3, 3])
        else:
            num_boosters = actuator_type.get_num_boosters()
            assert num_boosters != 0
            return spaces.Box(-1, +1, shape=(num_boosters,))

    def seed(self, seed=None):
        if seed is not None:
            self.random.seed(seed)
            self.config.seed(seed)

    def close(self):
        super().close()
        self.is_closed = True
        if self.window:
            window = self.window
            self.window = None
            self.canvas = None
            window.destroy()

    def step(self, action: Union[int, List[int]]):
        env_state = self.env_state
        rewards = env_state.apply_actions(action)
        observation = env_state.get_observation(self.bkg_color)
        infos = env_state.get_infos()
        done_states = env_state.get_done_states()
        return observation, rewards, done_states, infos

    def is_batched(self):
        return not self.squeeze or self.num_agents > 1

    def get_font(self):
        if self.cached_font is None:
            font_file_name = os.path.join(os.path.dirname(__file__), 'artifacts', 'ayar.ttf')
            self.cached_font = ImageFont.truetype(font_file_name, 12, encoding='unic')
        return self.cached_font

    def clear_caches(self):
        self.cached_font = None
        self.window = None
        self.canvas = None
        self.tk_agent_var = None

    def get_total_reward_of_selected_agent(self):
        agent_state = self.env_state.get_agent_state(self.selected_agent_index)
        return agent_state.cumulative_reward

    def get_agent_menu_var(self, agent_index: int):
        var = self.tk_agent_var
        if var is None:
            var = tkinter.IntVar(name='selected-agent')
            self.tk_agent_var = var
        if agent_index == self.selected_agent_index:
            var.set(agent_index)
        return var

    def get_agent_menu_command(self, agent_index: int):
        def _fn():
            var = self.tk_agent_var
            if var:
                var.set(agent_index)
                self.selected_agent_index = agent_index

        return _fn

    def create_agents_menu(self, menubar: tkinter.Menu):
        agents_menu = tkinter.Menu(menubar, tearoff=0)
        for i in range(self.num_agents):
            agents_menu.add_radiobutton(label=f'Agent {i}', variable=self.get_agent_menu_var(i), value=i,
                                        command=self.get_agent_menu_command(i))
        menubar.add_cascade(label='Agents', menu=agents_menu)

    def get_window_dimensions(self):
        ws = self.window_size
        if ws is not None:
            return ws
        window = self.window
        if window is None:
            return 0, 0,
        s_width = window.winfo_screenwidth()
        s_height = window.winfo_screenheight()
        min_dim = min(s_width, s_height)
        return int(min_dim * 0.95), int(min_dim * 0.85),

    def ensure_window_created(self):
        def _on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                self.close()

        if self.window is None:
            self.window = tkinter.Tk()
            self.window.title('FlatWorld')
            self.window.protocol("WM_DELETE_WINDOW", _on_closing)
            window_width, window_height = self.get_window_dimensions()
            self.window.geometry(f'{window_width}x{window_height}+0+0')
            self.window.bind('<KeyPress>', self.on_key_press)
            menubar = tkinter.Menu(self.window)
            if self.num_agents > 1:
                self.create_agents_menu(menubar)
            self.window.config(menu=menubar)
            self.canvas = tkinter.Canvas(self.window, width=window_width, height=window_height)
            self.canvas.pack()
        else:
            window_width, window_height = self.get_window_dimensions()
        return window_width, window_height

    def reset(self):
        self.bkg_color = self.config.get_background_color()
        self.min_cumulative_reward = self.config.get_min_cumulative_reward()
        if self.env_state and self.window:
            crpe = [None] * self.num_agents
            for i in range(self.num_agents):
                agent_state = self.env_state.get_agent_state(i)
                crpe[i] = agent_state.cumulative_reward
            self.cumulative_reward_prev_episode = crpe
        num_orientations = round(math.pi * 2 / self.config.get_rotation_step())
        orientation_helper = PrecalculatedOrientations(num_orientations, self.obs_resolution,
                                                       self.config.get_peripheral_vision_range())
        self.env_state = EnvState(self.config, self.random, orientation_helper, self.num_agents, self.obs_resolution,
                                  self.default_reward, self.reward_no_action, self.reward_toggle,
                                  self.reward_predator,
                                  self.min_cumulative_reward,
                                  self.predator_level,
                                  self.actuator_type,
                                  squeeze=self.squeeze)
        return self.env_state.get_observation(self.bkg_color)

    def render(self, mode='human'):
        """
        Note rgb_array return BGRA channels (so it can be used by VideoRecorder)
        """
        if mode == 'human':
            window_width, window_height = self.ensure_window_created()
            self.canvas.delete('all')
            pil_image = self.render_to_image(window_width, window_height, bkg_color='white')
            image = ImageTk.PhotoImage(image=pil_image)
            self.canvas.create_image(0, 0, image=image, anchor=NW)
            self.canvas.pack(side=TOP, expand=True, fill=BOTH)
            self.window.update()
            return None
        elif mode == 'rgba_array':
            image_width, image_height = self.window_size or (800, 600)
            pil_image = self.render_to_image(image_width, image_height, bkg_color='white')
            return np.asarray(pil_image)
        elif mode == 'rgb_array':
            image_width, image_height = self.window_size or (800, 600)
            pil_image = self.render_to_image(image_width, image_height, bkg_color='white')
            # gym's Monitor expects BGRA (not RGBA)
            return np.asarray(pil_image)[:, :, [2, 1, 0, 3]]
        else:
            raise Exception(f'mode {mode} not supported!')

    def render_to_image(self, width: int, height: int, bkg_color):
        pil_image = Image.new('RGBA', (width, height), color=bkg_color)
        draw = ImageDraw.Draw(pil_image)
        main_rect_dims = 4, 24, width - 4, height - 74,
        draw.rectangle(main_rect_dims, fill='black')
        main_x1, main_y1, main_x2, main_y2 = main_rect_dims
        projection_dims = main_x1, main_y2 + 6, main_x2, main_y2 + 36,
        depths_dims = main_x1, main_y2 + 42, main_x2, main_y2 + 72,
        draw.rectangle(projection_dims, fill='black')
        draw.rectangle(depths_dims, fill='black')
        selected_agent_state = self.env_state.get_agent_state(self.selected_agent_index)
        self.render_reward(self.selected_agent_index, selected_agent_state, draw, width)
        self.render_characters(draw, *main_rect_dims)
        self.render_elements(draw, *main_rect_dims)
        all_observations = self.env_state.get_observation(self.bkg_color)
        if self.is_batched():
            observation = all_observations[self.selected_agent_index]
        else:
            observation = all_observations
        self.render_projection(draw, observation, *projection_dims)
        self.render_depths(draw, observation, *depths_dims)
        return pil_image

    @staticmethod
    def tk_color(color: Tuple[int, int, int, int], background: Tuple[int, int, int, int]):
        bkg_r, bkg_g, bkg_b, _ = background
        color_r, color_g, color_b, alpha = color
        weight = alpha / 255.0
        new_r = round(color_r * weight + bkg_r * (1 - weight))
        new_g = round(color_g * weight + bkg_g * (1 - weight))
        new_b = round(color_b * weight + bkg_b * (1 - weight))
        mix = (new_r << 16) | (new_g << 8) | new_b
        return '#{n:06x}'.format(n=mix)

    @staticmethod
    def tk_color_simple(color: Tuple[int, int, int]):
        color_r, color_g, color_b = color
        mix = (color_r << 16) | (color_g << 8) | color_b
        return '#{n:06x}'.format(n=mix)

    @staticmethod
    def tk_color_simple_ia(color: Tuple[int, int, int, int]):
        color_r, color_g, color_b, _ = color
        mix = (color_r << 16) | (color_g << 8) | color_b
        return '#{n:06x}'.format(n=mix)

    def render_reward(self, agent_id: int, agent_state: AgentState, draw: ImageDraw.Draw, window_width: int):
        font = self.get_font()
        if self.cumulative_reward_prev_episode is not None:
            crpe = self.cumulative_reward_prev_episode[self.selected_agent_index]
            draw.text((4, 12), text='Prev episode: %.03f' % crpe, fill='red', font=font, anchor='lm')
        if self.num_agents > 1:
            done_text = '[DONE]' if agent_state.is_done else ''
            draw.text((window_width / 2 - 30, 12,), text=f'Agent {agent_id} {done_text}', fill='red',
                      font=font, anchor='lm')
        draw.text((window_width - 60, 12,), text='R: %.03f' % agent_state.cumulative_reward, fill='black',
                  font=font, anchor='lm')

    def render_depths(self, draw: ImageDraw.Draw, observation: np.ndarray, x1, y1, x2, y2):
        cell_width = (x2 - x1) / self.obs_resolution
        for i in range(self.obs_resolution):
            depth_as_color = round(observation[3, i])
            tk_color = '#{n:02x}{n:02x}{n:02x}'.format(n=depth_as_color)
            rx1 = x1 + cell_width * i
            rx2 = rx1 + cell_width
            draw.rectangle((rx1, y1, rx2, y2), fill=tk_color, outline='black')

    def render_projection(self, draw: ImageDraw.Draw, observation: np.ndarray, x1, y1, x2, y2):
        cell_width = (x2 - x1) / self.obs_resolution
        for i in range(self.obs_resolution):
            color = tuple(observation[:3, i])
            tkc = FlatWorldEnvironment.tk_color_simple(color)
            rx1 = x1 + cell_width * i
            rx2 = rx1 + cell_width
            draw.rectangle((rx1, y1, rx2, y2), fill=tkc, outline='black')

    def render_elements(self, draw: ImageDraw.Draw, x1, y1, x2, y2):
        background = 0, 0, 0, 255,
        g_w, g_h = self.env_state.grid_size
        scale_x, scale_y = (x2 - x1) / g_w, (y2 - y1) / g_h
        for element in self.env_state.get_elements():
            e_x, e_y = element.position
            rx1 = x1 + e_x * scale_x
            ry1 = y1 + e_y * scale_y
            rx2 = rx1 + scale_x
            ry2 = ry1 + scale_y
            draw.rectangle((rx1, ry1, rx2, ry2),
                           fill=FlatWorldEnvironment.tk_color(element.color, background),
                           outline='black')

    def render_characters(self, draw: ImageDraw.Draw, x1, y1, x2, y2):
        for p_state in self.env_state.get_character_states():
            if not p_state.is_done:
                self.render_character(p_state, draw, x1, y1, x2, y2)

    def render_character(self, character_state: BaseEnvCharacter, draw: ImageDraw.Draw, x1, y1, x2, y2):
        a_x, a_y = tuple(character_state.position)
        g_w, g_h = self.env_state.grid_size
        g_dim = min(g_w, g_h)
        render_dim = min(x2 - x1, y2 - y1)
        render_a_x = (x2 - x1) * a_x / g_w + x1
        render_a_y = (y2 - y1) * a_y / g_h + y1
        render_a_d = character_state.diameter * render_dim / g_dim
        render_a_r = render_a_d / 2
        aa = character_state.get_angle()
        aa_deg = np.rad2deg(aa)
        front_color = FlatWorldEnvironment.tk_color_simple_ia(character_state.get_front_color())
        back_color = FlatWorldEnvironment.tk_color_simple_ia(character_state.get_back_color())
        draw.pieslice((render_a_x - render_a_r, render_a_y - render_a_r,
                       render_a_x + render_a_r, render_a_y + render_a_r,),
                      start=aa_deg - 90, end=aa_deg + 90,
                      outline=front_color, fill=front_color)
        draw.pieslice((render_a_x - render_a_r, render_a_y - render_a_r,
                       render_a_x + render_a_r, render_a_y + render_a_r),
                      start=aa_deg + 90, end=aa_deg + 270,
                      outline=back_color, fill=back_color)
        eye_radius = character_state.eye_radius()
        if render_a_d / eye_radius > 8:
            eye_separation = math.pi / 10
            eye_color = character_state.eye_color()
            eye_1_x = render_a_x + render_a_r * 0.70 * math.cos(aa - eye_separation)
            eye_1_y = render_a_y + render_a_r * 0.70 * math.sin(aa - eye_separation)
            eye_2_x = render_a_x + render_a_r * 0.70 * math.cos(aa + eye_separation)
            eye_2_y = render_a_y + render_a_r * 0.70 * math.sin(aa + eye_separation)
            draw.ellipse((eye_1_x - eye_radius, eye_1_y - eye_radius,
                          eye_1_x + eye_radius, eye_1_y + eye_radius),
                         fill=eye_color)
            draw.ellipse((eye_2_x - eye_radius, eye_2_y - eye_radius,
                          eye_2_x + eye_radius, eye_2_y + eye_radius),
                         fill=eye_color)
        if character_state.has_tail():
            tail_xf = math.cos(aa + math.pi)
            tail_yf = math.sin(aa + math.pi)
            tail_x1 = render_a_x + render_a_r * tail_xf
            tail_y1 = render_a_y + render_a_r * tail_yf
            tail_x2 = render_a_x + render_a_r * 1.5 * tail_xf
            tail_y2 = render_a_y + render_a_r * 1.5 * tail_yf
            draw.line((tail_x1, tail_y1, tail_x2, tail_y2), fill='gray')

    def on_key_press(self, event):
        keysym = event.keysym
        if keysym == 'r' and (event.state & 0x04 != 0):
            self.reset()
            return
        at = self.actuator_type
        if at == ActuatorType.DEFAULT:
            action = FlatWorldEnvironment.key_press_to_default_action(keysym, event)
        elif at == ActuatorType.DUAL:
            action = FlatWorldEnvironment.key_press_to_dual_action(keysym, event)
        else:
            num_boosters = at.get_num_boosters()
            assert num_boosters != 0
            action = FlatWorldEnvironment.key_press_to_booster_action(keysym, event, num_boosters)
        self.keyboard_actions.append(action)

    @staticmethod
    def key_press_to_booster_action(keysym, event, num_boosters: int):
        assert num_boosters == 2 or num_boosters == 4
        action = np.zeros((num_boosters,))
        shift = (event.state & 0x1) != 0
        if keysym == '0' or keysym == 'KP_0':
            action[0] = 1
        elif keysym == '1' or keysym == 'KP_1':
            action[1] = 1
        elif keysym == 'Left' and not shift:
            action[0] = +1
            action[1] = -1
        elif keysym == 'Right' and not shift:
            action[0] = -1
            action[1] = +1
        elif keysym == 'Up':
            action[0] = -1
            action[1] = -1
        elif keysym == 'Down':
            action[0] = +1
            action[1] = +1
        if num_boosters == 4:
            if keysym == '2' or keysym == 'KP_2':
                action[2] = 1
            elif keysym == '3' or keysym == 'KP_3':
                action[3] = 1
            elif keysym == 'Left' and shift:
                action[2] = +1
                action[3] = +1
            elif keysym == 'Right' and shift:
                action[2] = -1
                action[3] = -1
        if event.state & 0x04 != 0:
            action = -action
        return action

    @staticmethod
    def key_press_to_default_action(keysym, event):
        action = ACTION_NONE
        if keysym == 'Up':
            action = ACTION_FORWARD
        elif keysym == 'Down':
            action = ACTION_BACK
        elif keysym == 'Left':
            action = ACTION_LEFT
        elif keysym == 'Right':
            action = ACTION_RIGHT
        elif keysym == 't' and (event.state & 0x04 != 0):
            action = ACTION_TOGGLE
        return action

    @staticmethod
    def key_press_to_dual_action(keysym, event):
        action = [0, 0]
        if keysym == 'Up':
            action = [1, 0]
        elif keysym == 'Down':
            action = [2, 0]
        elif keysym == 'Left':
            action = [0, 2]
        elif keysym == 'Right':
            action = [0, 1]
        return action

    def pull_next_keyboard_action(self) -> Union[int, List[int]]:
        actions = self.keyboard_actions
        if self.actuator_type == ActuatorType.DEFAULT:
            default_action = ACTION_NONE
        elif self.actuator_type == ActuatorType.DUAL:
            default_action = [0, 0]
        else:
            default_action = np.zeros((self.actuator_type.get_num_boosters(),))
        if len(actions) == 0:
            action = default_action
        else:
            action = actions[0]
            self.keyboard_actions = actions[1:]
        result = action
        if self.is_batched():
            result = [default_action] * self.num_agents
            result[self.selected_agent_index] = action
        return result
