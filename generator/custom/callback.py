import os
import numpy as np

from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

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
        self.save_path = os.path.join(log_dir, 'model')
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
            results_df = load_results(self.log_dir)
            x, y = ts2xy(results_df, "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                sol_length_mean = np.mean(results_df['sol-length'][-100:])
                feasibility_mean = np.mean(results_df['sol-length'][-100:] > 0)
                crates_mean = np.mean(results_df['crate'][-100:])
                solver_mean = np.mean(results_df['solver'][-100:])

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
                if self.num_timesteps % 20000 == 0:
                    curr_model_path = os.path.join(self.save_path, str(self.num_timesteps))
                    print(f"Saving latest model to {curr_model_path}")
                    self.model.save(curr_model_path)

                # save episode reward mean
                self.kwargs['wandb_session'].log(data={'ep_rew_mean': mean_reward,
                                                       'sol-length_mean': sol_length_mean,
                                                       'feasibility': feasibility_mean,
                                                       'crates_mean': crates_mean,
                                                       'solver_mean': solver_mean},
                                                 step=self.num_timesteps)
                # print("Episode reward: {:.2f}".format(mean_reward))

        return True