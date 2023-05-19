import gym
import numpy as np
import time


def test_the_solver(agent, env_name, eval_num, display=False, level=None, opt_steps=None):
    solved = []
    rewards = []

    if level is None:
        raise ValueError('No level is provided for testing the agent.')

    env = gym.make(env_name, level=level)
    env._max_episode_steps = 5 * opt_steps if opt_steps is not None else env._max_episode_steps
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
                env.render()
                time.sleep(1)
            state, reward, done, _ = env.step(action.item())
            episode_reward += reward

            i += 1
        if display:
            print('current state\n')
            print(env.room_state)
            print('The game is over, the episode reward is {}'.format(episode_reward))
        if i < env.max_steps:
            solved.append(1)
            if display:
                print('Solved the game...')
                time.sleep(1)

        rewards.append(episode_reward)
    env.close()

    return np.sum(solved) / eval_num, np.sum(rewards) / eval_num
