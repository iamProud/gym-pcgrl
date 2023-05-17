import os

from stable_baselines3 import PPO
from utils import get_exp_name, max_exp_idx, load_model, make_vec_envs
from generator.custom.callback import SaveOnBestTrainingRewardCallback


def train_generator(game, representation, experiment, steps, n_cpu, **kwargs):
    env_name = '{}-{}-v0'.format(game, representation)
    exp_name = get_exp_name(game, representation, experiment, **kwargs)
    resume = kwargs.get('resume', False)

    n = max_exp_idx(exp_name) + 1
    model = None

    if resume and n > 1:
        model = load_model(f'runs/{exp_name}_{n-1}_log/generator/model')

    log_dir = f'runs/{exp_name}_{n}_log/generator'
    os.makedirs(log_dir)

    kwargs = {
        **kwargs,
        'render_rank': 0
    }

    env = make_vec_envs(env_name, representation, log_dir, n_cpu, **kwargs)

    if not resume or model is None:
        model = PPO('MlpPolicy', env, verbose=1)
    else:
        model.set_env(env)

    check_freq = kwargs.get('check_freq', 10000) / n_cpu
    model.learn(
        total_timesteps=int(steps),
        callback=SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_dir, verbose=2, kwargs=kwargs)
    )
