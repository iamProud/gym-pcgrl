import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from solver.custom.callback import SaveOnBestTrainingRewardCallback
from utils import get_exp_name, max_exp_idx, load_model


def train_solver(env_name, policy, timesteps, n_cpu, **kwargs):
    experiment = kwargs.get('experiment', None)
    exp_name = get_exp_name('arl-sokoban', 'turtle', experiment)
    n = max_exp_idx(exp_name)
    resume = kwargs.get('resume', False)

    model = None

    if resume and n > 1:
        model = load_model(f'runs/{exp_name}_{n-1}_log/solver/model')

    log_dir = f'runs/{exp_name}_{n}_log/solver'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    env = make_vec_env(env_name, n_envs=n_cpu, monitor_dir=log_dir,
                       env_kwargs={'generator_path': kwargs.get('generator_path', None),
                                   'infer_kwargs': kwargs.get('infer_kwargs', None)})

    if not resume or model is None:
        model = PPO(policy, env, verbose=1)
    else:
        model.set_env(env)

    check_freq = kwargs.get('eval_freq', 10000) / n_cpu
    model.learn(total_timesteps=timesteps, callback=SaveOnBestTrainingRewardCallback(check_freq, log_dir, kwargs=kwargs))
