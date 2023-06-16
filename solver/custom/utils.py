from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

def make_env(env_name, rank=0, log_dir=None, **kwargs):
    '''
    Return a function that will initialize the environment when called.
    '''
    def _thunk():
        env = make_vec_env(env_name, n_envs=1, monitor_dir=log_dir,
                           monitor_kwargs= {'info_keywords': kwargs.get('info_keywords', ())},
                           env_kwargs={'generator_path': kwargs.get('generator_path', None),
                                       'infer_kwargs': kwargs.get('infer_kwargs', None),
                                       'max_steps': kwargs.get('max_steps', 200),
                                       'env_id': rank})

        return env
    return _thunk

def make_vec_envs(env_name, log_dir, n_cpu, **kwargs):
    '''
    Prepare a vectorized environment using a list of 'make_env' functions.
    '''
    if n_cpu > 1:
        env_lst = []
        for i in range(n_cpu):
            env_lst.append(make_env(env_name, i, log_dir, **kwargs))
        env = SubprocVecEnv(env_lst)
    else:
        env = DummyVecEnv([make_env(env_name, 0, log_dir, **kwargs)])
    return env
