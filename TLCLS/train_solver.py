import os.path

import gym
import TLCLS.gym_sokoban

from TLCLS.common.fix_and_reinit import fix_and_reinit
from TLCLS.common.train_the_agent import train_the_agent
from TLCLS.common.ActorCritic import ActorCritic
from TLCLS.common.RolloutStorage import RolloutStorage
from TLCLS.common.multiprocessing_env import SubprocVecEnv

import torch
import torch.autograd as autograd
import torch.optim as optim

import argparse
import wandb

def train_solver(args, wandb_session, log_dir, training_levels, model_path=None):
    source_env_name = 'Curriculum-Sokoban-v2'
    source_task_map = training_levels

    #source task training
    def make_env():
        def _thunk():
            env = gym.make(source_env_name, data_path = source_task_map)
            return env
        return _thunk

    envs = [make_env() for i in range(args.num_envs)]
    envs = SubprocVecEnv(envs)
    state_shape = (3, 80, 80)
    num_actions = 5

    actor_critic = ActorCritic(state_shape, num_actions=num_actions)

    if model_path is not None:
        device = 'cuda' if args.USE_CUDA and torch.cuda.is_available() else 'cpu'
        actor_critic.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        actor_critic.eval()

    rollout = RolloutStorage(args.rolloutStorage_size, args.num_envs, state_shape)
    optimizer = optim.RMSprop(actor_critic.parameters(), lr=args.lr, eps=args.eps, alpha=args.alpha)

    if args.USE_CUDA:
        if not torch.cuda.is_available():
            raise ValueError('You wanna use cuda, but the machine you are on doesnt support')
        elif torch.cuda.is_available():
            Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda()
            actor_critic.cuda()
            rollout.cuda()
    else:
        Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs)


    print('Train the solving agent...')
    train_the_agent(envs, args.num_envs, source_env_name, Variable, state_shape, actor_critic, optimizer, rollout, data_path=source_task_map, args=args, wandb_session=wandb_session) #train and save the model;

    #save the model
    os.mkdir(log_dir+'/solver_model')
    torch.save(actor_critic.state_dict(), os.path.join(log_dir, 'solver_model', 'model.pkl'))
