import os
from PIL import Image
import numpy as np
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_certain_tile, calc_num_regions
from gym_pcgrl.envs.probs.pushing.engine import State, BFSAgent #, AStarAgent
from globals import *

class PushingProblem(Problem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        super().__init__()
        self._width = 5
        self._height = 5
        self._prob = {"empty": 0.30, "solid": 0.64, "player": 0.01, "crate": 0.02, "target": 0.01, "trap": 0.02}
        self._border_tile = "solid"

        self._solver_power = 5000

        self._max_crates = 5
        self._max_traps = 5

        self._target_solution = 10

        self._rewards = {
            "empty": -0.2,
            "player": 3,
            "crate": 2,
            "target": 3,
            "trap": 2,
            "regions": 5,
            "ratio": 0,
            "dist-win": 0.0,
            "sol-length": 1
        }

    """
    Get a list of all the different tile names
    Returns:
        string[]: that contains all the tile names
    """
    def get_tile_types(self):
        return ["empty", "solid", "player", "crate", "target", "trap"]

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
            "empty": calc_certain_tile(map_locations, ["empty"]),
            "player": calc_certain_tile(map_locations, ["player"]),
            "target": calc_certain_tile(map_locations, ["target"]),
            "crate": calc_certain_tile(map_locations, ["crate"]),
            "trap": calc_certain_tile(map_locations, ["trap"]),
            "regions": calc_num_regions(map, map_locations, ["empty", "player", "target", "crate"]),
            "dist-win": self._width * self._height * (self._width + self._height),
            "solution": []
        }

        if map_stats["player"] == 1 and map_stats["regions"] == 1 and map_stats["target"] == 1:
                map_stats["dist-win"], map_stats["solution"] = self._run_game(map)
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
        rewards = {
            "empty": get_range_reward(new_stats["empty"], old_stats["empty"], 0, np.inf),
            "player": get_range_reward(new_stats["player"], old_stats["player"], 1, 1),
            "target": get_range_reward(new_stats["target"], old_stats["target"], 1, 1),
            "crate": get_range_reward(new_stats["crate"], old_stats["crate"], 1, self._max_crates),
            "trap": get_range_reward(new_stats["trap"], old_stats["trap"], 0, self._max_traps),
            "regions": get_range_reward(new_stats["regions"], old_stats["regions"], 1, 1),
            "dist-win": get_range_reward(new_stats["dist-win"], old_stats["dist-win"], -np.inf, -np.inf),
            "sol-length": get_range_reward(len(new_stats["solution"]), len(old_stats["solution"]), np.inf, np.inf)
        }
        #calculate the total reward
        return rewards["crate"] * self._rewards["crate"] +\
            rewards["trap"] * self._rewards["trap"] +\
            rewards["player"] * self._rewards["player"] +\
            rewards["target"] * self._rewards["target"] +\
            rewards["regions"] * self._rewards["regions"] +\
            rewards["dist-win"] * self._rewards["dist-win"] +\
            rewards["sol-length"] * self._rewards["sol-length"] +\
            rewards["empty"] * self._rewards["empty"]


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
        return len(new_stats["solution"]) >= self._target_solution


    """
    Private function that runs the game on the input level

    Parameters:
        map (string[][]): the input level to run the game on

    Returns:
        float: how close you are to winning (0 if you win)
        int: the solution length if you win (0 otherwise)
    """
    def _run_game(self, map):
        gameCharacters=" #@$xo"
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

        state = State()
        state.stringInitialize(lvlString.split("\n"))

        # aStarAgent = AStarAgent()
        bfsAgent = BFSAgent()

        sol,solState,iters = bfsAgent.getSolution(state, self._solver_power)
        if solState.checkWin():
            return 0, sol
        # sol,solState,iters = aStarAgent.getSolution(state, 1, self._solver_power)
        # if solState.checkWin():
        #     return 0, sol
        # sol,solState,iters = aStarAgent.getSolution(state, 0.5, self._solver_power)
        # if solState.checkWin():
        #     return 0, sol
        # sol,solState,iters = aStarAgent.getSolution(state, 0, self._solver_power)
        # if solState.checkWin():
        #     return 0, sol
        return solState.getHeuristic(), []


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
            "target": new_stats["target"],
            "crate": new_stats["crate"],
            "trap": new_stats["trap"],
            "regions": new_stats["regions"],
            "dist-win": new_stats["dist-win"],
            "sol-length": len(new_stats["solution"])
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
                "empty": Image.open(os.path.dirname(__file__) + "/pushing/empty.png").convert('RGBA'),
                "solid": Image.open(os.path.dirname(__file__) + "/pushing/solid.png").convert('RGBA'),
                "player": Image.open(os.path.dirname(__file__) + "/pushing/player.png").convert('RGBA'),
                "crate": Image.open(os.path.dirname(__file__) + "/pushing/crate.png").convert('RGBA'),
                "target": Image.open(os.path.dirname(__file__) + "/pushing/target.png").convert('RGBA'),
                "trap": Image.open(os.path.dirname(__file__) + "/pushing/trap.png").convert('RGBA')
            }
        return super().render(map)
