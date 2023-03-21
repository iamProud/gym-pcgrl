"""
Run a trained agent and get generated maps
"""
import model
from stable_baselines import PPO2

import time
from utils import make_vec_envs

def infer(game, representation, model_path, **kwargs):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    env_name = '{}-{}-v0'.format(game, representation)
    if game == "binary":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
        kwargs['cropped_size'] = 22
    elif game == "sokoban" or game == "robosoko":
        model.FullyConvPolicy = model.FullyConvPolicySmallMap
        kwargs['cropped_size'] = 16
    kwargs['render'] = True

    agent = PPO2.load(model_path)
    env = make_vec_envs(env_name, representation, None, 1, **kwargs)

    for j in range(kwargs.get('num_executions', 1)):
        obs = env.reset()
        dones = False

        for i in range(kwargs.get('trials', 1)):
            while not dones:
                action, _ = agent.predict(obs)
                obs, _, dones, info = env.step(action)
                if kwargs.get('verbose', False):
                    print(info[0])
                if dones:
                    break
            time.sleep(0.2)

################################## MAIN ########################################
game = 'sokoban'
representation = 'turtle'

model_path = 'runs/{}_{}_5_log/best_model.pkl'.format(game, representation)
kwargs = {
    'change_percentage': 0.5,
    'trials': 1,
    'verbose': True,
    'num_executions': 100,
    'solver_power': 20000,
    'width': 8,
    'height': 8,
    'rewards': {'player': 6},
    'probs': {'empty': 0.4, 'solid': 0.56, 'player': 0, 'crate': 0.04, 'target': 0.04},
    'max_crates': 1,
    'max_targets': 1,
    'min_solution': 20
}

if __name__ == '__main__':
    infer(game, representation, model_path, **kwargs)

    input("Press enter to exit!")