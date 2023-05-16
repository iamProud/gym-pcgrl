import os

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from solver.custom.callback import SaveOnBestTrainingRewardCallback
from utils import get_exp_name, max_exp_idx, load_model
from stable_baselines3.common.vec_env import SubprocVecEnv

from solver.custom.customCNN import CustomCNN


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

    env = make_vec_env(env_name, n_envs=n_cpu, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv,
                       monitor_kwargs={'info_keywords': kwargs.get('info_keywords', ())},
                       env_kwargs={'generator_path': kwargs.get('generator_path', None),
                                   'infer_kwargs': kwargs.get('infer_kwargs', None),
                                   'max_steps': kwargs.get('max_steps', 200),
                                   'level_repetitions': kwargs.get('level_repetitions', 1)
                                    })

    num_actions = env.action_space.n
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[dict(pi=[512, num_actions], vf=[512, 1])],
    )

    if not resume or model is None:
        model = A2C(policy, env, policy_kwargs=policy_kwargs, verbose=1)
    else:
        model.set_env(env)

    check_freq = kwargs.get('eval_freq', 10000) / n_cpu
    model.learn(total_timesteps=timesteps, callback=SaveOnBestTrainingRewardCallback(check_freq, log_dir, kwargs=kwargs))
