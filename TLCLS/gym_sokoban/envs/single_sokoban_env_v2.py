import numpy as np
import math
from .sokoban_env import SokobanEnv
from .render_utils import room_to_rgb
import os
from os import listdir
from os.path import isfile, join
import requests
import zipfile
from tqdm import tqdm
import random
import numpy as np


class SingleSokobanEnv_v2(SokobanEnv):
    def __init__(self, level, max_steps=120):
        super(SingleSokobanEnv_v2, self).__init__(reset=False)

        self.max_steps = max_steps
        self.level = level.split('\n')[:-1]

    def reset(self):

        if self.level is None:
            raise ValueError("Missing level")

        self.select_room()

        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0
        self.dim_room = np.array(self.room_state).shape

        starting_observation = room_to_rgb(self.room_state, self.room_fixed)

        return starting_observation

    def generate_room(self, select_map):
        room_fixed = []
        room_state = []

        targets = []
        boxes = []
        for row in select_map:
            room_f = []
            room_s = []

            for e in row:
                if e == '#':
                    room_f.append(0)
                    room_s.append(0)

                elif e == '@':
                    self.player_position = np.array([len(room_fixed), len(room_f)])
                    room_f.append(1)
                    room_s.append(5)


                elif e == '$':
                    boxes.append((len(room_fixed), len(room_f)))
                    room_f.append(1)
                    room_s.append(4)

                elif e == '.':
                    targets.append((len(room_fixed), len(room_f)))
                    room_f.append(2)
                    room_s.append(2)

                elif e == ' ':
                    room_f.append(1)
                    room_s.append(1)

            room_fixed.append(room_f)
            room_state.append(room_s)

        # used for replay in room generation, unused here because pre-generated levels
        box_mapping = {}
        num_boxes = len(boxes)

        return np.array(room_fixed), np.array(room_state), box_mapping, num_boxes

    def select_room(self):
        self.room_fixed, self.room_state, self.box_mapping, self.num_boxes = self.generate_room(self.level)
