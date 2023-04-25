"""
Run a trained agent and get generated maps
"""
from stable_baselines3 import PPO
from stable_baselines3.common.policies import obs_as_tensor

import time
from utils import make_vec_envs
from globals import *

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

    env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    agent = PPO.load(model_path) #, custom_objects={'observation_space': env.observation_space})

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
                    if info[0]['sol-length'] > 0:
                        generated += 1
                    break
            # time.sleep(0.2)

    env.close()

################################## MAIN ########################################
game = 'sokoban_tlcls'
representation = 'turtle'
run_idx = 1

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
    'solver_power': 10000
}

game_path = f'shared_runs/{kwargs["width"]}x{kwargs["height"]}/sokoban'
run_path = f'{game_path}/sokoban_{representation}_{run_idx}_log/'
# model_path = 'runs/{}_{}_1_log/best_model.zip'.format(game, representation)
# model_path = run_path + '/best_model.zip'
model_path = run_path + '/57000000.zip'

if __name__ == '__main__':
    infer(game, representation, model_path, **kwargs)

    # input("Press enter to exit!")