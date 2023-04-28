from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.helper import get_int_prob, get_string_map
import numpy as np
import gym
from gym import spaces
import PIL
from PIL import Image
import os
from gym_pcgrl.envs.helper import safe_map
import json
import traceback
import glob
import re

def get_exp_name(game, representation, experiment, **kwargs):
    exp_name = '{}_{}'.format(game, representation)
    if experiment is not None:
        exp_name = '{}_{}'.format(exp_name, experiment)
    return exp_name

def max_exp_idx(exp_name):
    log_dir = os.path.join("./runs", exp_name)
    log_files = glob.glob('{}*'.format(log_dir))
    if len(log_files) == 0:
        n = 0
    else:
        log_ns = [re.search(r'{}_(\d+)'.format(exp_name), f).group(1) for f in log_files]
        log_ns = [int(n) for n in log_ns]
        n = max(log_ns)
    return int(n)

"""
The PCGRL GYM Environment
"""
class PcgrlEnv(gym.Env):
    """
    The type of supported rendering
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    """
    Constructor for the interface.

    Parameters:
        prob (string): the current problem. This name has to be defined in PROBLEMS
        constant in gym_pcgrl.envs.probs.__init__.py file
        rep (string): the current representation. This name has to be defined in REPRESENTATIONS
        constant in gym_pcgrl.envs.reps.__init__.py
    """
    def __init__(self, prob="binary", rep="narrow"):
        self._prob = PROBLEMS[prob]()
        self._rep = REPRESENTATIONS[rep]()
        self._rep_stats = None
        self._iteration = 0
        self._changes = 0
        self._max_changes = max(int(0.2 * self._prob._width * self._prob._height), 1)
        self._max_iterations = self._max_changes * self._prob._width * self._prob._height
        self._heatmap = np.zeros((self._prob._height, self._prob._width))

        self.seed()
        self.viewer = None
        self.render_mode = 'rgb_array'

        # generated images/environments
        stack = traceback.extract_stack()
        self.is_inference = any('infer' in s for s in stack)

        if self.is_inference:
            experiment_name = get_exp_name(prob, rep, 5)
            experiment_idx = max_exp_idx(experiment_name)
            self.path_generated = os.path.join('runs', f'{experiment_name}_{experiment_idx}_log', 'generated')
            if not os.path.exists(self.path_generated):
                os.makedirs(self.path_generated)

                with open(self.path_generated + '/info.json', 'w') as f:
                    json.dump({'trials': 0, 'success-rate': 0, 'avg-sol-length': 0, 'avg-crates': 0, 'avg-free-percent': 0,
                               'failed': {
                                   'total': 0, '0-player': 0, '2+players': 0, 'region': 0, 'crate-target': 0
                               }}, f)

        self.action_space = self._rep.get_action_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space = self._rep.get_observation_space(self._prob._width, self._prob._height,
                                                                 self.get_num_tiles())
        self.observation_space.spaces['heatmap'] = spaces.Box(low=0, high=self._max_changes, dtype=np.uint8,
                                                              shape=(self._prob._height, self._prob._width))

    """
    Seeding the used random variable to get the same result. If the seed is None,
    it will seed it with random start.

    Parameters:
        seed (int): the starting seed, if it is None a random seed number is used.

    Returns:
        int[]: An array of 1 element (the used seed)
    """
    def seed(self, seed=None):
        seed = self._rep.seed(seed)
        self._prob.seed(seed)
        return [seed]

    """
    Resets the environment to the start state

    Returns:
        Observation: the current starting observation have structure defined by
        the Observation Space
    """
    def reset(self):
        self._changes = 0
        self._iteration = 0
        self._rep.reset(self._prob._width, self._prob._height,
                        get_int_prob(self._prob._prob, self._prob.get_tile_types()))
        self._rep_stats = self._prob.get_stats(get_string_map(self._rep._map, self._prob.get_tile_types()))
        self._prob.reset(self._rep_stats)
        self._heatmap = np.zeros((self._prob._height, self._prob._width))

        observation = self._rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()
        return observation

    """
    Get the border tile that can be used for padding

    Returns:
        int: the tile number that can be used for padding
    """
    def get_border_tile(self):
        return self._prob.get_tile_types().index(self._prob._border_tile)

    """
    Get the number of different type of tiles that are allowed in the observation

    Returns:
        int: the number of different tiles
    """
    def get_num_tiles(self):
        return len(self._prob.get_tile_types())

    """
    Adjust the used parameters by the problem or representation

    Parameters:
        change_percentage (float): a value between 0 and 1 that determine the
        percentage of tiles the algorithm is allowed to modify. Having small
        values encourage the agent to learn to react to the input screen.
        **kwargs (dict(string,any)): the defined parameters depend on the used
        representation and the used problem
    """
    def adjust_param(self, **kwargs):
        if 'change_percentage' in kwargs:
            percentage = min(1, max(0, kwargs.get('change_percentage')))
            self._max_changes = max(int(percentage * self._prob._width * self._prob._height), 1)
        self._max_iterations = self._max_changes * self._prob._width * self._prob._height
        self._prob.adjust_param(**kwargs)
        self._rep.adjust_param(**kwargs)
        self.action_space = self._rep.get_action_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space = self._rep.get_observation_space(self._prob._width, self._prob._height,
                                                                 self.get_num_tiles())
        self.observation_space.spaces['heatmap'] = spaces.Box(low=0, high=self._max_changes, dtype=np.uint8,
                                                              shape=(self._prob._height, self._prob._width))
        self.render_mode = kwargs.get('render_mode', 'rgb_array')

    """
    Advance the environment using a specific action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        observation: the current observation after applying the action
        float: the reward that happened because of applying that action
        boolean: if the problem eneded (episode is over)
        dictionary: debug information that might be useful to understand what's happening
    """
    def step(self, action):
        self._iteration += 1
        # save copy of the old stats to calculate the reward
        old_stats = self._rep_stats
        # update the current state to the new state based on the taken action
        change, x, y = self._rep.update(action)
        if change > 0:
            self._changes += change
            self._heatmap[y][x] += 1.0
            self._rep_stats = self._prob.get_stats(get_string_map(self._rep._map, self._prob.get_tile_types()))
        # calculate the values
        observation = self._rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()
        reward = self._prob.get_reward(self._rep_stats, old_stats)
        done = self._prob.get_episode_over(self._rep_stats,
                                           old_stats) or self._changes >= self._max_changes or self._iteration >= self._max_iterations
        info = self._prob.get_debug_info(self._rep_stats, old_stats)
        info["iterations"] = self._iteration
        info["changes"] = self._changes
        info["max_iterations"] = self._max_iterations
        info["max_changes"] = self._max_changes
        # return the values

        if done and self.is_inference:
            self.log_inference(info)

        return observation, reward, done, info

    """
    Logs the results of the inference to a file
    """
    def log_inference(self, info):
        with open(self.path_generated + "/info.json", "r") as f:
            data = json.load(f)
            data["trials"] += 1
            successful = data['success-rate'] * (data['trials'] - 1)
            data['success-rate'] = (successful + 1) / (data['trials']) if (info["sol-length"] > 0) else successful / (
            data['trials'])
        with open(self.path_generated + "/info.json", "w") as f:
            json.dump(data, f)

        if info["sol-length"] > 0:
            self.log_successful(info, successful)
        else:
            self.log_failed(info)
            self.render(mode=self.render_mode)

    """
    Logs the results of the successful inference to a file
    """
    def log_successful(self, info, successful):
        with open(self.path_generated + "/info.json", "r") as f:
            data = json.load(f)

            data['avg-sol-length'] = (data['avg-sol-length'] * successful + info["sol-length"]) / (successful + 1)
            data['avg-crates'] = (data['avg-crates'] * successful + info["crate"]) / (successful + 1)
            free_ratio = np.count_nonzero(self._rep._map == 0) / (self._prob._width * self._prob._height)
            data['avg-free-percent'] = (data['avg-free-percent'] * successful + free_ratio) / (successful + 1)
        with open(self.path_generated + "/info.json", "w") as f:
            json.dump(data, f)

        # get file number
        listdir = os.listdir(self.path_generated)

        if len(listdir) == 0:
            file_count = 0
        else:
            file_count = -1
            for generated_file in listdir:
                try:
                    val = int(generated_file.split('.')[0])
                    file_count = max(file_count, val)
                except ValueError:
                    continue

            file_count += 1

        # save map as image
        img = self.render(mode='rgb_array')
        img.save(f'{self.path_generated}/{file_count}.jpeg')

        # save map as .txt
        final_map = np.pad(self._rep._map, 1, constant_values=1)
        safe_map(final_map, self._rep_stats['solution'], self.path_generated, file_count)

    """
    Logs the results of the failed inference to a file
    failed obj: 'total', '0-player', '2+players', 'region', 'crate-target'
    """
    def log_failed(self, info):
        with open(self.path_generated + "/info.json", "r") as f:
            data = json.load(f)
            data['failed']['total'] += 1
            data['failed']['0-player'] += 1 if info['player'] == 0 else 0
            data['failed']['2+players'] += 1 if info['player'] >= 2 else 0
            data['failed']['region'] += 1 if info['regions'] > 1 else 0
            data['failed']['crate-target'] += 1 if info['crate'] != info['target'] else 0
        with open(self.path_generated + "/info.json", "w") as f:
            json.dump(data, f)


    """
    Render the current state of the environment

    Parameters:
        mode (string): the value has to be defined in render.modes in metadata

    Returns:
        img or boolean: img for rgb_array rendering and boolean for human rendering
    """

    def render(self, mode='human'):
        tile_size = 16
        img = self._prob.render(get_string_map(self._rep._map, self._prob.get_tile_types()))
        img = self._rep.render(img, self._prob._tile_size, self._prob._border_size).convert("RGB")
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            if not hasattr(img, 'shape'):
                img = np.array(img)
            self.viewer.imshow(img)
            return self.viewer.isopen

    """
    Close the environment
    """
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
