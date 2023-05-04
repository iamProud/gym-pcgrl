"""
Run a trained agent and get generated maps
"""
from stable_baselines3 import PPO
from stable_baselines3.common.policies import obs_as_tensor

import time
from utils import make_vec_envs

def predict_probability(model, state):
    action_keys = ['left', 'right', 'up', 'down', 'empty', 'wall', 'player', 'crate', 'goal']
    obs = obs_as_tensor(state, model.policy.device)
    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    probs_np = probs.detach().numpy()[0]
    probs_dict = dict(zip(action_keys, probs_np))

    return probs_dict, max(probs_dict, key=probs_dict.get)

def infer(game, representation, model_path, **kwargs):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    env_name = f'{game}-{representation}-v0'
    kwargs['render'] = kwargs.get('render', True)
    min_solution = kwargs.get('min_solution', 1)
    env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    agent = PPO.load(model_path) #, custom_objects={'observation_space': env.observation_space})
    kwargs['solver_max_solved'] = kwargs.get('solver_max_solved', 1)

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
                # probs, best = predict_probability(agent, obs)
                # print('Probabilities:', probs, 'Best:', best)
                obs, rewards, dones, info = env.step(action)
                # print("Action: " + str(action) + ", reward: " + str(rewards))
                if kwargs.get('verbose', False):
                    print(info[0])
                if dones:
                    if info[0]['sol-length'] >= min_solution and \
                            (info[0]['solver'] is None or info[0]['solver'] < kwargs['solver_max_solved']):
                        generated += 1
                    break
            # time.sleep(0.2)

    env.close()

################################## MAIN ########################################
game = 'sokoban_solver'
representation = 'turtle'
run_idx = 8

kwargs = {
    'change_percentage': 0.5,
    'trials': 1,
    # 'verbose': True,
    'num_executions': 1,
    'render': True,
    'width': 5,
    'height': 5,
    'cropped_size': 10,
    'probs': {"empty": 0.6, "solid": 0.34, "player": 0.02, "crate": 0.02, "target": 0.02},
    'min_solution': 15,
    'max_crates': 2,
    'max_targets': 2,
    'solver_power': 10000,
    'solver_max_solved': None,
}

game_path = f'shared_runs/{kwargs["width"]}x{kwargs["height"]}/sokoban'
run_path = f'{game_path}/sokoban_{representation}_{run_idx}_5_log/'
# model_path = run_path + 'pcg_model/best_model.zip'
model_path = 'shared_runs/5x5/sokoban/sokoban_solver_turtle_8_5_log/pcg_model/best_model.zip'
if __name__ == '__main__':
    infer(game, representation, model_path, **kwargs)

    # input("Press enter to exit!")
