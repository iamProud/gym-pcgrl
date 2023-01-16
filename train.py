#Install stable-baselines as described in the documentation

import os

# import tensorflow as tf
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, EventCallback

# from model import FullyConvPolicyBigMap, FullyConvPolicySmallMap, CustomPolicyBigMap, CustomPolicySmallMap
from utils import get_exp_name, max_exp_idx, load_model, make_vec_envs
from globals import *

def eval_feasibility(model, kwargs):
    """
    Evaluate the feasibility of the model. This is done by running the current best model.

    :param model: (PPO) The current best model to evaluate.
    :param kwargs: (dict) The kwargs to pass to the environment.
    """
    env = make_vec_envs(f'{game}-{representation}-v0', representation, None, 1, **kwargs)

    total_runs = 100
    feasible_runs = 0

    for i in range(total_runs):
        obs = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs)
            obs, rewards, done, info = env.step(action)

            if done:
                if info[0]['sol-length'] > 0:
                    feasible_runs += 1
                break

    env.close()
    return feasible_runs / total_runs

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward.

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1, kwargs: dict = {}):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.kwargs = kwargs

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                      print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
                    feasibility = eval_feasibility(self.model, self.kwargs)
                    print("Feasibility: {:.2f}".format(feasibility))
                    self.logger.record('feasibility', feasibility)

        return True


def main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs):
    env_name = '{}-{}-v0'.format(game, representation)
    exp_name = get_exp_name(game, representation, experiment, **kwargs)
    resume = kwargs.get('resume', False)

    if game == "binary":
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        kwargs['cropped_size'] = 10
    n = max_exp_idx(exp_name)

    if not resume:
        n = n + 1
    log_dir = 'runs/{}_{}_{}'.format(exp_name, n, 'log')
    if not resume:
        os.mkdir(log_dir)
    else:
        model = load_model(log_dir)
    kwargs = {
        **kwargs,
        'render_rank': 0,
        'render': render,
    }

    env = make_vec_envs(env_name, representation, log_dir, n_cpu, **kwargs)
    if not resume or model is None:
        policy = "MlpPolicy"
        model = PPO(policy, env, verbose=1, tensorboard_log="./runs")
    else:
        model.set_env(env)
    if not logging:
        model.learn(total_timesteps=int(steps), tb_log_name=exp_name)
    else:
        model.learn(total_timesteps=int(steps), tb_log_name=exp_name, callback=SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, verbose=2, kwargs=kwargs))

################################## MAIN ########################################
experiment = None
steps = 1e6
logging = True
n_cpu = 50
kwargs = {
    'resume': False
}

if __name__ == '__main__':
    main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs)
