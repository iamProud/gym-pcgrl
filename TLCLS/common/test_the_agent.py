import gym
import TLCLS.gym_sokoban

import torch
import numpy as np
import time

from TLCLS.common.utils import hwc2chw

def test_the_agent(agent, env_name, data_path, USE_CUDA, eval_num, display=False, level=None, device='cuda'):
    #every test will test all sub-cases that the agent was training on;

    solved = []
    rewards = []
    
    #specify the environment you wanna use; v0 means sample sub-cases randomly, and v1 only sample targeted sub-cases;
    if level is None:
        env = gym.make(env_name, data_path=data_path)
    else:
        env = gym.make(env_name, level=level)

    for i in range(eval_num):

        episode_reward = 0

        state = env.reset()
        if display:
            print('current state\n')
            print(env.room_state)
        state = hwc2chw(state, test=True)
        if USE_CUDA:
            state = state.cuda(device=device)
        action = agent.select_action(state.unsqueeze(0), test=1)
        if display:
            print('action selected: {}'.format(action.item()))
            time.sleep(1)
        next_state, reward, done, _ = env.step(action.item())
        episode_reward += reward
        next_state = hwc2chw(next_state, test=True)

        i = 1

        while not done:
            state = next_state
            if USE_CUDA:
                state = state.cuda(device=device)
            with torch.no_grad():
                action = agent.select_action(state.unsqueeze(0), test=1)
            if display:
                print('current state\n')
                print(env.room_state)
                print('action selected: {}'.format(action.item()))
                env.render()
                time.sleep(1)
            next_state, reward, done, _ = env.step(action.item())
            episode_reward += reward
            next_state = hwc2chw(next_state, test=True)

            i += 1
        if display:
            print('current state\n')
            print(env.room_state)
            print('The game is over, the episode rewrd is {}'.format(episode_reward))
        if i < env.max_steps:
            solved.append(1)
            if display:
                print('Solved the game...')
                time.sleep(1)
        
        rewards.append(episode_reward)
    env.close()

    return np.sum(solved)/eval_num, np.sum(rewards)/eval_num
