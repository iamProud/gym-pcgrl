import os
import wandb
import numpy as np
import gym
import gym_pcgrl
import gym_sokoban

from utils import get_exp_name, max_exp_idx
from generator.train import train_generator
from solver.train import train_solver

################################## MAIN ########################################
game = 'arl-sokoban'
representation = 'turtle'
experiment = 11
experiment_name = get_exp_name(game, representation, experiment)

steps = 1e5
n_cpu = 1

adversarial_learning = {
    'enabled': True,
    'iterations': 10,
    'generator_iterations': steps,
    'solver_iterations': 1e5,
}

generator_kwargs = {
    'resume': True,
    'render': False,
    'render_mode': 'rgb_array',
    'info_keywords': ('sol-length', 'crate', 'solver'),
    'change_percentage': 0.2,
    'width': 5,
    'height': 5,
    'cropped_size': 10,
    'probs': {"empty": 0.45, "solid": 0.4, "player": 0.05, "crate": 0.05, "target": 0.05},
    'min_solution': 1,
    'max_crates': 2,
    'solver_power': 5000,
    'solver_path': "runs/arl-sokoban_turtle_11_1_log/solver/model/best_model",
    'infer_solver_max_solved': np.inf
}

solver_kwargs = {
    'experiment': experiment,
    'num_steps': adversarial_learning['solver_iterations'],
    'num_envs': 1,
    'eval_freq': 1000,
    'eval_num': 20,
    'generator_path': None,
    'infer_kwargs': None
}

wand_mode = 'disabled'
start_with_solver = True

if __name__ == '__main__':
    if adversarial_learning['enabled']:
        for i in range(2, adversarial_learning['iterations']+1):
            wandb_pcg_session = wandb.init(project=f'arlpcg-{game}', config=generator_kwargs,
                               name=f'{experiment_name}-{i}', group='generator', mode=wand_mode)
            generator_kwargs['wandb_session'] = wandb_pcg_session

            if i == 1:
                generator_kwargs['max_crates'] = 1

            if not start_with_solver:
                train_generator(game, representation, experiment, steps, n_cpu, **generator_kwargs)
            start_with_solver = False

            experiment_idx = max_exp_idx(experiment_name)
            log_dir = os.path.join('runs', f'{experiment_name}_{experiment_idx}_log')

            infer_kwargs = generator_kwargs.copy()
            infer_kwargs['change_percentage'] = 0.5

            if i > 1:
                infer_kwargs['infer_solver_max_solved'] = 1

            if i == 1:
                infer_kwargs['infer_max_solution'] = 5
                infer_kwargs['infer_max_crates'] = 1

            wandb_pcg_session.finish()

            solver_kwargs['generator_path'] = os.path.join(log_dir, 'generator', 'model', 'best_model')
            solver_kwargs['infer_kwargs'] = infer_kwargs
            wandb_solver_session = wandb.init(project=f'arlpcg-{game}', config=solver_kwargs,
                                              name=f'{experiment_name}-{i}', reinit=True, group='solver',
                                              mode=wand_mode)

            config = wandb.config

            train_solver('Sokoban-arl-v0', 'CnnPolicy', solver_kwargs['num_steps'], solver_kwargs['num_envs'], **solver_kwargs)

            generator_kwargs['solver_path'] = os.path.join('runs', f'{experiment_name}_{experiment_idx}_log', 'solver', 'model', 'best_model')

            wandb_solver_session.finish()

    else:
        wandb_pcg_session = wandb.init(project=f'pcg-{game}', config=generator_kwargs,
                                name=f'{experiment_name}', group='generator', mode=wand_mode)
        generator_kwargs['wandb_session'] = wandb_pcg_session
        train_generator(game, representation, experiment, steps, n_cpu, **generator_kwargs)

        wandb_pcg_session.finish()
