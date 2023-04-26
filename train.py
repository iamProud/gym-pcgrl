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

import wandb
from wandb.integration.sb3 import WandbCallback
from inference import infer
import argparse
from TLCLS.train_solver import train_solver
from TLCLS.transform import transform_map

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
        self.save_path = os.path.join(log_dir, "pcg_model")
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
                    self.model.save(self.save_path+'/best_model')

                # Save current model
                if self.num_timesteps % 100000 == 0:
                    curr_model_path = os.path.join(self.save_path, str(self.num_timesteps))
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


def main(game, representation, experiment, steps, n_cpu, logging, **kwargs):
    env_name = '{}-{}-v0'.format(game, representation)
    exp_name = get_exp_name(game, representation, experiment, **kwargs)
    resume = kwargs.get('resume', False)

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
        'render_rank': 0
    }

    env = make_vec_envs(env_name, representation, log_dir, n_cpu, **kwargs)

    if not resume or model is None:
        model = PPO(policy, env, verbose=1, tensorboard_log="./runs", device=device)
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
game = 'sokoban_solver'
representation = 'turtle'
policy = 'MlpPolicy'
device='auto'
experiment = 3
steps = 5e6
logging = True
n_cpu = 50
mode_GAN = {
    'enabled': True,
    'iterations': 20,
    'generator_iterations': steps,
    'generate_levels': 20,
    'solver_iterations': 1e6,
}


kwargs = {
    'resume': False,
    'render': False,
    'render_mode': 'rgb_array',
    'change_percentage': 0.2,
    'width': 5,
    'height': 5,
    'cropped_size': 10,
    'probs': {"empty": 0.6, "solid": 0.34, "player": 0.02, "crate": 0.02, "target": 0.02},
    'min_solution': 15,
    'max_crates': 2,
    'max_targets': 2,
    'solver_power': 10000,
    'num_level_generation': mode_GAN['generate_levels'],
    'solver_path': None
}

experiment_name = get_exp_name(game, representation, experiment, **kwargs)

# wandb pcg hyperparameters
wandb_hyperparameter = dict(
    policy=policy,
    game=game,
    representation=representation,
    size=f'{kwargs["width"]}x{kwargs["height"]}',
    change_percentage=kwargs['change_percentage'],
    prob_empty=kwargs['probs']['empty'],
    prob_solid=kwargs['probs']['solid'],
    prob_player=kwargs['probs']['player'],
    prob_crate=kwargs['probs']['crate'],
    prob_target=kwargs['probs']['target'],
    target_solution=kwargs['min_solution'],
    max_crates=kwargs['max_crates'],
)

# solver hyperparameters
description = 'TLCLS'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--exp_name', type=str, default=experiment_name)
parser.add_argument('--num_steps', type=int, default=mode_GAN['solver_iterations'])
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--entropy_coef', type=float, default=0.1)
parser.add_argument('--value_loss_coef', type=float, default=0.5)
parser.add_argument('--max_grad_norm', type=float, default=0.5)
parser.add_argument('--rolloutStorage_size', type=int, default=5)
parser.add_argument('--num_envs', type=int, default=30)
parser.add_argument('--eval_freq', type=int, default=1000)
parser.add_argument('--eval_num', type=int, default=20)
parser.add_argument('--lr', type=float, default=7e-4)
parser.add_argument('--eps', type=float, default=1e-5)
parser.add_argument('--alpha', type=float, default=0.99)
solver_args = parser.parse_args()
solver_args.USE_CUDA = False

if __name__ == '__main__':
    wandb_pcg_session = wandb.init(project=f'pcgrl-{game}', config=wandb_hyperparameter,
                               name=experiment_name, mode='online')
    kwargs['wandb_session'] = wandb_pcg_session

    if mode_GAN['enabled']:
        for i in range(1, mode_GAN['iterations']+1):
            main(game, representation, experiment, steps, n_cpu, logging, **kwargs)
            experiment_idx = max_exp_idx(experiment_name)

            # generate new environments
            log_dir = os.path.join('runs', f'{experiment_name}_{experiment_idx}_log')
            best_model = os.path.join(log_dir, 'pcg_model', 'best_model.zip')
            infer_kwargs = kwargs.copy()
            infer_kwargs['change_percentage'] = 0.5

            print("Start inference")
            infer(game, representation, best_model, **infer_kwargs)

            maps_folder = os.path.join(log_dir, 'generated')
            os.mkdir(log_dir+'/transformed')
            for filename in os.listdir(maps_folder):
                if filename.endswith(".txt"):
                    transform_map(log_dir, filename)

            wandb_solver_session = wandb.init(project=f'pcgrl-{game}-solver', config=vars(solver_args), name=experiment_name, reinit=True,
                                       group=f'{experiment_name}_{experiment_idx}', mode='online')

            config = wandb.config

            last_solver = None
            if experiment_idx > 1:
                last_solver = os.path.join('runs', f'{experiment_name}_{experiment_idx-1}_log', 'model.pkl')
            train_solver(solver_args, wandb_solver_session, log_dir, last_solver)

            kwargs['solver_path'] = os.path.join('runs', f'{experiment_name}_{experiment_idx}_log', 'solver_model', 'model.pkl')

            wandb_solver_session.finish()

    else:
        main(game, representation, experiment, steps, n_cpu, logging, **kwargs)

    wandb_pcg_session.finish()
