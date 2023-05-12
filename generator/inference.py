"""
Run a trained agent and get generated maps
"""
import numpy as np

from stable_baselines3 import PPO
from utils import make_vec_envs


def is_desired_solution(info, **kwargs):
    return info['sol-length'] >= kwargs.get('min_solution', 1) and \
        (info['solver'] is None or info['solver'] < kwargs.get('infer_solver_max_solved'), np.inf) and \
        kwargs.get('infer_min_crate', 1) <= info.get('crate') <= kwargs.get('infer_max_crate', np.inf) and \
        kwargs.get('infer_min_solution', 1) <= info.get('sol-length') <= kwargs.get('infer_max_solution', np.inf)


def infer(game, representation, model_path, **kwargs):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    env_name = f'{game}-{representation}-v0'
    kwargs['render'] = kwargs.get('render', True)

    env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    agent = PPO.load(model_path)

    i = 0
    generated = 0
    while True:
        if kwargs.get('num_level_generation', None) is not None:
            if generated >= kwargs.get('num_level_generation'):
                break
        else:
            if i >= kwargs.get('num_executions', 1):
                break

        i += 1
        obs = env.reset()
        dones = False

        for i in range(kwargs.get('trials', 1)):
            while not dones:
                action, _ = agent.predict(obs)

                obs, rewards, dones, info = env.step(action)

                if kwargs.get('verbose', False):
                    print(info[0])
                if dones:
                    # print('INFER-info: sol-length=', info[0]['sol-length'], '| solver=', info[0]['solver'])
                    if is_desired_solution(info[0], **kwargs):
                        generated += 1
                    break
            # time.sleep(0.2)

    env.close()

    return info[0]['map']

################################## MAIN ########################################
game = 'sokoban_solver'
representation = 'turtle'
experiment = 10

kwargs = {
    'change_percentage': 0.5,
    'trials': 1,
    # 'verbose': True,
    'num_executions': 1,
    'render': False,
    'width': 5,
    'height': 5,
    'cropped_size': 10,
    'probs': {"empty": 0.45, "solid": 0.4, "player": 0.05, "crate": 0.05, "target": 0.05},
    'min_solution': 1,
    'max_crates': 2,
    'solver_power': 10000,
    # only save levels with the following properties
    # 'solver_max_solved': np.inf,
    # 'infer_min_crate': 1,
    # 'infer_max_crate': 2,
    # 'infer_max_solution': 2,
    'num_level_generation': 100,
    'solver': 'shared_runs/5x5/sokoban/sokoban_solver_turtle_10_7_log/solver_model/model.zip'
}

game_path = f'shared_runs/{kwargs["width"]}x{kwargs["height"]}/sokoban'
run_path = f'{game_path}/sokoban_solver_{representation}_{experiment}_7_log/'
model_path = run_path + 'pcg_model/best_model.zip'
# model_path = 'shared_runs/5x5/sokoban/sokoban_solver_turtle_8_5_log/pcg_model/best_model.zip'
if __name__ == '__main__':
    infer(game, representation, model_path, **kwargs)

    # input("Press enter to exit!")
