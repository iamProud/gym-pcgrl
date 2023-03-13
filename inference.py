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
    kwargs['cropped_size'] = config['cropped_size']
    kwargs['render'] = kwargs.get('render', True)

    agent = PPO.load(model_path)
    env = make_vec_envs(env_name, representation, None, 1, **kwargs)

    for j in range(kwargs.get('num_executions', 1)):
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
                    break
            time.sleep(0.2)

################################## MAIN ########################################
# model_path = 'runs/{}_{}_1_log/best_model.zip'.format(game, representation)
model_path = run_path + '/best_model.zip'
kwargs = {
    'change_percentage': 0.5,
    'trials': 1,
    # 'verbose': True,
    'num_executions': 100,
    'render': render,
}

if __name__ == '__main__':
    infer(game, representation, model_path, **kwargs)

    # input("Press enter to exit!")