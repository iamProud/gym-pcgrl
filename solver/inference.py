import os
import gym
import numpy as np
import time

from stable_baselines3 import PPO
from gym_sokoban.envs.helper import transform_map
from gym_sokoban.envs.render_utils import room_to_rgb
from PIL import Image


def test_the_solver(agent, env_name, eval_num, display=False, level=None, optimal_solution=None, data_path=None):
    solved = []
    rewards = []

    if data_path is not None:
        env = gym.make(env_name, data_path=data_path)
    else:
        env = gym.make(env_name, level=level)

    env._max_episode_steps = optimal_solution['mult'] * optimal_solution['steps'] if optimal_solution is not None else env._max_episode_steps
    for i in range(eval_num):

        episode_reward = 0

        state = env.reset()
        done = False
        i = 1

        while not done:
            action, _ = agent.predict(state)

            if display:
                print('current state\n')
                print(env.room_state)
                print('action selected: {}'.format(action.item()))
                # Image.fromarray(room_to_rgb(env.room_state)).save(f'../shared_runs/5x5/solve-example/2/{i}.png')
                time.sleep(1)
            state, reward, done, _ = env.step(action.item())
            episode_reward += reward

            i += 1
        if display:
            print('current state\n')
            print(env.room_state)
            # Image.fromarray(room_to_rgb(env.room_state)).save(f'../shared_runs/5x5/solve-example/2/{i}.png')
            print('The game is over, the episode reward is {}'.format(episode_reward))
        if i < env.max_steps:
            solved.append(1)
            if display:
                print('Solved the game...')
                time.sleep(1)

        rewards.append(episode_reward)
    env.close()

    return np.sum(solved) / eval_num, np.sum(rewards) / eval_num


### Test the solver
level = '../shared_runs/5x5/generation-example/1/level.txt'
model_path = '../shared_runs/5x5/sokoban/arl-sokoban_turtle_3_10_log/solver/model/best_model.zip'

if __name__ == '__main__':
    model = PPO.load(model_path)

    with open(level, 'r') as f:
        map_file = open(level, 'r')
        map_lines = map_file.readlines()

        sokoban_level = transform_map(map_lines)
        sokoban_level = ''.join(sokoban_level)


    test_the_solver(agent=model,
                    env_name='Single-Sokoban-v0',
                    eval_num=1,
                    display=True,
                    level=sokoban_level,
                    optimal_solution={'mult': 1, 'steps': 200}
                    )