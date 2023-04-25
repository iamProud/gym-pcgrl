import gym_pcgrl.envs
from gym_pcgrl.envs.probs.binary_prob import BinaryProblem
from gym_pcgrl.envs.probs.ddave_prob import DDaveProblem
from gym_pcgrl.envs.probs.mdungeon_prob import MDungeonProblem
from gym_pcgrl.envs.probs.sokoban_prob import SokobanProblem
from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
from gym_pcgrl.envs.probs.smb_prob import SMBProblem
from gym_pcgrl.envs.probs.pushing_prob import PushingProblem
from gym_pcgrl.envs.probs.sokoban_tlcls_prob import SokobanTlclsProblem
from gym_pcgrl.envs.probs.sokoban_solver_prob import SokobanSolverProblem

# all the problems should be defined here with its corresponding class
PROBLEMS = {
    "binary": BinaryProblem,
    "ddave": DDaveProblem,
    "mdungeon": MDungeonProblem,
    "sokoban": SokobanProblem,
    "zelda": ZeldaProblem,
    "smb": SMBProblem,
    "pushing": PushingProblem,
    "sokoban_tlcls": SokobanTlclsProblem,
    "sokoban_solver": SokobanSolverProblem
}
