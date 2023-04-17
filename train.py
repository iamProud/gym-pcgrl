#Install stable-baselines as described in the documentation

import os
from torch import nn

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

# from model import FullyConvPolicyBigMap, FullyConvPolicySmallMap, CustomPolicyBigMap, CustomPolicySmallMap
from model import CustomCNNPolicy
from utils import get_exp_name, max_exp_idx, load_model, make_vec_envs, eval_feasibility
from globals import *

import wandb
from wandb.integration.sb3 import WandbCallback

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
        self.kwargs['change_percentage'] = 1

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

                # Save current model
                if self.num_timesteps % 100000 == 0:
                    curr_model_path = os.path.join(self.log_dir, str(self.num_timesteps))
                    print(f"Saving latest model to {curr_model_path}")
                    self.model.save(curr_model_path)

                    # Evaluate the feasibility of the current model
                    env = make_vec_envs(f'{game}-{representation}-v0', representation, None, 1, **self.kwargs)
                    feasibility = eval_feasibility(env, self.model, 100)
                    # print("Feasibility: {:.2f}".format(feasibility))
                    self.kwargs['wandb_session'].log(data={'feasibility': feasibility}, step=self.num_timesteps)
                    env.close

                # save episode reward mean
                self.kwargs['wandb_session'].log(data={'ep_rew_mean': mean_reward}, step=self.num_timesteps)
                # print("Episode reward: {:.2f}".format(mean_reward))

        return True


def main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs):
    env_name = '{}-{}-v0'.format(game, representation)
    exp_name = get_exp_name(game, representation, experiment, **kwargs)
    resume = kwargs.get('resume', False)

    kwargs['cropped_size'] = config['cropped_size']
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
        policy_kwargs = None

        if policy == 'CnnPolicy':
            policy_kwargs = dict(
                activation_fn=nn.ReLU,
                net_arch=[512],
                features_extractor_class=CustomCNNPolicy,
                features_extractor_kwargs=dict(features_dim=512),
            )
        model = PPO(policy, env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./runs", device=device)
    else:
        model.set_env(env)
    if not logging:
        model.learn(total_timesteps=int(steps), tb_log_name=exp_name)
    else:
        check_freq = 10000 / n_cpu
        model.learn(
            total_timesteps=int(steps),
            tb_log_name=exp_name,
            callback=SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_dir, verbose=2, kwargs=kwargs)
        )

################################## MAIN ########################################
policy = 'MlpPolicy'
experiment = None
steps = 1e8
logging = True
n_cpu = 50
experiment = run_idx
exp_name = get_exp_name(game, representation, experiment)

# wandb hyperparameters
wandb_hyperparameter = dict(
    policy=policy,
    game=game,
    representation=representation,
    size=f'{config["width"]}x{config["height"]}',
    change_percentage=config['change_percentage'],
    prob_empty=config['probabilities']['empty'],
    prob_solid=config['probabilities']['solid'],
    prob_player=config['probabilities']['player'],
    prob_crate=config['probabilities']['crate'],
    prob_target=config['probabilities']['target'],
    target_solution=config['target_solution'],
    max_crates=config['max_crates'],
)

kwargs = {
    'resume': False,
    'change_percentage': config['change_percentage'],
    'width': config['width'],
    'height': config['height'],
    'cropped_size': config['cropped_size'],
    'probs': config['probabilities'],
    'min_solution': config['target_solution'],
    'max_crates': config['max_crates'],
    'max_targets': config['max_crates'],
    'solver_power': config['solver_power']
}

if __name__ == '__main__':
    wandb_session = wandb.init(project=f'pcgrl-{game}', config=wandb_hyperparameter, name=exp_name, mode='online')
    kwargs['wandb_session'] = wandb_session

    main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs)

    wandb_session.finish()
