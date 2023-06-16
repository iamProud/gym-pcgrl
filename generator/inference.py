"""
Run a trained agent and get generated maps
"""
from stable_baselines3 import PPO
from utils import make_vec_envs
from generator.custom.log import *


def infer(game, representation, model_path, device='auto', **kwargs):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    env_name = f'{game}-{representation}-v0'
    kwargs['render'] = kwargs.get('render', True)

    env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    agent = PPO.load(model_path, device=device)

    path_generated = kwargs.get('path_generated', './generated')
    if not os.path.exists(path_generated):
        os.makedirs(path_generated)

        with open(path_generated + '/info.json', 'w') as f:
            json.dump({'trials': 0, 'success-rate': 0, 'avg-sol-length': 0, 'avg-crates': 0, 'avg-free-percent': 0,
                       'failed': {
                           'total': 0, '0-player': 0, '2+players': 0, 'region': 0, 'crate-target': 0
                       }}, f)

    i = 0
    generated = 0
    generated_maps = []

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
                    success = log_inference(info[0], **kwargs)
                    # print('INFER-info: sol-length=', info[0]['sol-length'], '| solver=', info[0]['solver'])
                    if success:
                        generated += 1
                        generated_maps.append({'map': info[0]['map'], 'sol-length': info[0]['sol-length']})
                    break
            # time.sleep(0.2)

    env.close()

    return generated_maps

################################## MAIN ########################################
game = 'sokoban'
representation = 'turtle'
experiment = 1

kwargs = {
    'log_json': True,
    # 'change_percentage': 0.5,
    'trials': 1,
    # 'verbose': True,
    'num_executions': 1,
    'render': True,
    'width': 5,
    'height': 5,
    'cropped_size': 10,
    'probs': {"empty": 0.45, "solid": 0.4, "player": 0.05, "crate": 0.05, "target": 0.05},
    'solver_power': 5000,
    'min_solution': 1,
    # only save levels with the following properties
    'infer': {
        # 'crate': {'max': 1},
        'sol-length': {'min': 1},
        # 'solver': {'min': 0}
    },
    'num_level_generation': 100,
    # 'solver': 'shared_runs/5x5/sokoban/sokoban_solver_turtle_11_10_log/solver_model/model.pkl',
    'path_generated': 'foo'
}

# game_path = f'shared_runs/{kwargs["width"]}x{kwargs["height"]}/sokoban/arl-sokoban_{representation}_{experiment}_10_log/'
game_path = f'shared_runs/{kwargs["width"]}x{kwargs["height"]}/sokoban/{game}_{representation}_{experiment}_log/'
# run_path = f'runs/arl-sokoban_{representation}_{experiment}_10_log/'
model_path = game_path + 'best_model.zip'
# model_path = game_path + 'generator/model/latest_model.zip'
# model_path = 'shared_runs/5x5/sokoban/sokoban_solver_turtle_8_5_log/pcg_model/best_model.zip'
if __name__ == '__main__':
    infer(game, representation, model_path, **kwargs)

    # input("Press enter to exit!")
