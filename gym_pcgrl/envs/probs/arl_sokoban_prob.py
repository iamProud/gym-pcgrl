import os
import numpy as np
import torch
from PIL import Image

from stable_baselines3 import PPO
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_certain_tile, calc_num_regions
from gym_pcgrl.envs.probs.sokoban.engine import State,BFSAgent,AStarAgent
from solver.inference import test_the_solver

"""
Generate a fully connected Sokoban(https://en.wikipedia.org/wiki/Sokoban) level that can be solved
"""
class ArlSokobanProblem(Problem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        super().__init__()
        self._ID = None
        self._solver_path = None
        self.solver_agent = None
        self.current_map = None
        self._width = 5
        self._height = 5
        self._prob = {"empty":0.30, "solid":0.55, "player": 0.01, "crate": 0.07, "target": 0.07}
        self._border_tile = "solid"

        self._solver_power = 20000

        self._max_crates = 5
        self._max_solution = np.inf
        self._target_solution = 30

        self._rewards = {
            "player": 5,
            "crate": 2,
            "target": 2,
            "regions": 5,
            "ratio": 2,
            "dist-win": 0.0,
            "sol-length": 1,
            "solver": 20
        }

    """
    Get a list of all the different tile names

    Returns:
        string[]: that contains all the tile names
    """
    def get_tile_types(self):
        return ["empty", "solid", "player", "crate", "target"]

    """
    Adjust the parameters for the current problem

    Parameters:
        width (int): change the width of the problem level
        height (int): change the height of the problem level
        probs (dict(string, float)): change the probability of each tile
        intiialization, the names are "empty", "solid", "player", "crate", "target"
        max_crates or max_targets (int): the max number of crates or target both
        suppose to be the same value so setting one is enough
        target_solution (int): the target solution length that the level is considered a success if reached
        rewards (dict(string,float)): the weights of each reward change between the new_stats and old_stats
    """
    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)

        self._solver_path = kwargs.get('solver_path', self._solver_path)
        if self._solver_path is not None and self.solver_agent is None:
            self._ID = (kwargs['env_id'] % (torch.cuda.device_count()-1)) + 1
            device = f"cuda:{self._ID}" if torch.cuda.is_available() else "cpu"

            self.solver_agent = PPO.load(self._solver_path, device=device)

        self._solver_power = kwargs.get('solver_power', self._solver_power)
        self._max_crates = kwargs.get('max_crates', self._max_crates)
        self._max_solution = kwargs.get('max_solution', self._max_solution)

        self._target_solution = kwargs.get('min_solution', self._target_solution)

        rewards = kwargs.get('rewards')
        if rewards is not None:
            for t in rewards:
                if t in self._rewards:
                    self._rewards[t] = rewards[t]

    """
    Private function that runs the game on the input level

    Parameters:
        map (string[][]): the input level to run the game on

    Returns:
        float: how close you are to winning (0 if you win)
        int: the solution length if you win (0 otherwise)
    """
    def _run_game(self, map):
        gameCharacters=" #@$."
        string_to_char = dict((s, gameCharacters[i]) for i, s in enumerate(self.get_tile_types()))
        lvlString = ""
        for x in range(self._width+2):
            lvlString += "#"
        lvlString += "\n"
        for i in range(len(map)):
            for j in range(len(map[i])):
                string = map[i][j]
                if j == 0:
                    lvlString += "#"
                lvlString += string_to_char[string]
                if j == self._width-1:
                    lvlString += "#\n"
        for x in range(self._width+2):
            lvlString += "#"
        lvlString += "\n"

        self.current_map = lvlString

        state = State()
        state.stringInitialize(lvlString.split("\n"))

        aStarAgent = AStarAgent()
        bfsAgent = BFSAgent()

        sol,solState,iters = bfsAgent.getSolution(state, self._solver_power)
        if solState.checkWin():
            return 0, sol
        sol,solState,iters = aStarAgent.getSolution(state, 1, self._solver_power)
        if solState.checkWin():
            return 0, sol
        sol,solState,iters = aStarAgent.getSolution(state, 0.5, self._solver_power)
        if solState.checkWin():
            return 0, sol
        sol,solState,iters = aStarAgent.getSolution(state, 0, self._solver_power)
        if solState.checkWin():
            return 0, sol
        return solState.getHeuristic(), []

    """
    Get the current stats of the map

    Returns:
        dict(string,any): stats of the current map to be used in the reward, episode_over, debug_info calculations.
        The used status are "player": number of player tiles, "crate": number of crate tiles,
        "target": number of target tiles, "reigons": number of connected empty tiles,
        "dist-win": how close to the win state, "sol-length": length of the solution to win the level
    """
    def get_stats(self, map):
        map_locations = get_tile_locations(map, self.get_tile_types())
        map_stats = {
            "player": calc_certain_tile(map_locations, ["player"]),
            "crate": calc_certain_tile(map_locations, ["crate"]),
            "target": calc_certain_tile(map_locations, ["target"]),
            "regions": calc_num_regions(map, map_locations, ["empty","player","crate","target"]),
            "dist-win": self._width * self._height * (self._width + self._height),
            "solution": [],
            "solver": 0
        }
        if map_stats["player"] == 1 and map_stats["crate"] == map_stats["target"] and map_stats["crate"] > 0 and map_stats["regions"] == 1:
                map_stats["dist-win"], map_stats["solution"] = self._run_game(map)

                if len(map_stats["solution"]) > 0 and self._solver_path is not None:
                    avg_solved, reward_mean = test_the_solver(agent=self.solver_agent, env_name='Single-Sokoban-v0',
                                                             eval_num=20, display=False, level=self.current_map)
                    print_args = (self._ID, avg_solved, round(reward_mean, 2), map_stats["crate"], len(map_stats["solution"]))
                    print('AGENT ID: {0:<3}  avg_solved: {1:<4}  reward_mean: {2:=4}  crates: {3:<2}  sol-length: {4:<3}'.format(*print_args))
                    map_stats["solver"] = avg_solved
        return map_stats

    """
    Get the current game reward between two stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        float: the current reward due to the change between the old map stats and the new map stats
    """
    def get_reward(self, new_stats, old_stats):
        #longer path is rewarded and less number of regions is rewarded
        rewards = {
            "player": get_range_reward(new_stats["player"], old_stats["player"], 1, 1),
            "crate": get_range_reward(new_stats["crate"], old_stats["crate"], 1, self._max_crates),
            "target": get_range_reward(new_stats["target"], old_stats["target"], 1, self._max_crates),
            "regions": get_range_reward(new_stats["regions"], old_stats["regions"], 1, 1),
            "ratio": get_range_reward(abs(new_stats["crate"]-new_stats["target"]), abs(old_stats["crate"]-old_stats["target"]), -np.inf, -np.inf),
            "dist-win": get_range_reward(new_stats["dist-win"], old_stats["dist-win"], -np.inf, -np.inf),
            "sol-length": get_range_reward(len(new_stats["solution"]), len(old_stats["solution"]), np.inf, self._max_solution),
            "solver": get_range_reward(new_stats["solver"], old_stats["solver"], 0.5, 0.5)
        }
        #calculate the total reward
        return rewards["player"] * self._rewards["player"] +\
            rewards["crate"] * self._rewards["crate"] +\
            rewards["target"] * self._rewards["target"] +\
            rewards["regions"] * self._rewards["regions"] +\
            rewards["ratio"] * self._rewards["ratio"] +\
            rewards["dist-win"] * self._rewards["dist-win"] +\
            rewards["sol-length"] * self._rewards["sol-length"] +\
            rewards["solver"] * self._rewards["solver"]

    """
    Uses the stats to check if the problem ended (episode_over) which means reached
    a satisfying quality based on the stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        boolean: True if the level reached satisfying quality based on the stats and False otherwise
    """
    def get_episode_over(self, new_stats, old_stats):
        return len(new_stats["solution"]) >= self._target_solution #and \
            #(self._solver_path is None or (new_stats["solver"] > 0 and new_stats["solver"] < 1))

    """
    Get any debug information need to be printed

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        dict(any,any): is a debug information that can be used to debug what is
        happening in the problem
    """
    def get_debug_info(self, new_stats, old_stats):
        return {
            "player": new_stats["player"],
            "crate": new_stats["crate"],
            "target": new_stats["target"],
            "regions": new_stats["regions"],
            "dist-win": new_stats["dist-win"],
            "sol-length": len(new_stats["solution"]),
            "solver": new_stats["solver"]
        }

    """
    Get an image on how the map will look like for a specific map

    Parameters:
        map (string[][]): the current game map

    Returns:
        Image: a pillow image on how the map will look like using sokoban graphics
    """
    def render(self, map):
        if self._graphics == None:
            self._graphics = {
                "empty": Image.open(os.path.dirname(__file__) + "/sokoban/empty.png").convert('RGBA'),
                "solid": Image.open(os.path.dirname(__file__) + "/sokoban/solid.png").convert('RGBA'),
                "player": Image.open(os.path.dirname(__file__) + "/sokoban/player.png").convert('RGBA'),
                "crate": Image.open(os.path.dirname(__file__) + "/sokoban/crate.png").convert('RGBA'),
                "target": Image.open(os.path.dirname(__file__) + "/sokoban/target.png").convert('RGBA')
            }
        return super().render(map)
