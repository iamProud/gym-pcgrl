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
experiment = 1
experiment_name = get_exp_name(game, representation, experiment)

steps = 4e5
n_cpu = 20

adversarial_learning = {
    'enabled': True,
    'iterations': 10,
    'generator_iterations': steps,
    'solver_iterations': 2e5,
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
    'max_crates': 1,
    'solver_power': 5000,
    # 'solver_path': "runs/arl-sokoban_turtle_11_1_log/solver/model/best_model",
    'infer_solver_max_solved': np.inf
}

solver_kwargs = {
    'experiment': experiment,
    'info_keywords': ('all_boxes_on_target',),
    'num_steps': adversarial_learning['solver_iterations'],
    'num_envs': 20,
    'eval_freq': 10000,
    'generator_path': None,
    'infer_kwargs': None
}

wand_mode = 'online'
start_with_solver = True

if __name__ == '__main__':
    if adversarial_learning['enabled']:
        for i in range(1, adversarial_learning['iterations']+1):
            if not start_with_solver:
                wandb_pcg_session = wandb.init(project=f'arlpcg-{game}', config=generator_kwargs,
                                               name=f'{experiment_name}-{i}', group='generator', mode=wand_mode)
                generator_kwargs['wandb_session'] = wandb_pcg_session

                train_generator(game, representation, experiment, steps, n_cpu, **generator_kwargs)

                wandb_pcg_session.finish()

            start_with_solver = False

            experiment_idx = max_exp_idx(experiment_name)
            log_dir = os.path.join('runs', f'{experiment_name}_{experiment_idx}_log')

            infer_kwargs = generator_kwargs.copy()
            infer_kwargs['change_percentage'] = 0.5

            if i > 1:
                infer_kwargs['infer_solver_max_solved'] = 1
                solver_kwargs['max_steps'] = 200

            if i == 1:
                infer_kwargs['infer_max_solution'] = 2
                infer_kwargs['infer_max_crates'] = 1

                solver_kwargs['max_steps'] = 100


            solver_kwargs['generator_path'] = os.path.join(log_dir, 'generator', 'model', 'best_model')
            solver_kwargs['infer_kwargs'] = infer_kwargs
            wandb_solver_session = wandb.init(project=f'arlpcg-{game}', config=solver_kwargs,
                                              name=f'{experiment_name}-{i}', reinit=True, group='solver',
                                              mode=wand_mode)

            solver_kwargs['wandb_session'] = wandb_solver_session

            train_solver(env_name='Sokoban-arl-v0', policy='CnnPolicy', timesteps=solver_kwargs['num_steps'], n_cpu=solver_kwargs['num_envs'], **solver_kwargs)

            generator_kwargs['solver_path'] = os.path.join('runs', f'{experiment_name}_{experiment_idx}_log', 'solver', 'model', 'best_model')

            wandb_solver_session.finish()

    else:
        wandb_pcg_session = wandb.init(project=f'pcg-{game}', config=generator_kwargs,
                                name=f'{experiment_name}', group='generator', mode=wand_mode)
        generator_kwargs['wandb_session'] = wandb_pcg_session
        train_generator(game, representation, experiment, steps, n_cpu, **generator_kwargs)

        wandb_pcg_session.finish()
