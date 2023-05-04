import gym
import TLCLS.gym_sokoban

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pickle

from TLCLS.common.utils import hwc2chw
from TLCLS.common.test_the_agent import test_the_agent

def train_the_agent(envs, num_envs, env_name, Variable, state_shape, actor_critic, optimizer, rollout, data_path, args, wandb_session):

    state = envs.reset()
    #import ipdb; ipdb.set_trace()
    state = hwc2chw(state)
    current_state = torch.FloatTensor(np.float32(state))
    rollout.states[0].copy_(current_state)
    
    i_step = 0

    #if the agent should be trained until the performance plateau, we just simply set the number of training step to a really large number, which is 1e8.
    while i_step < args.num_steps:
        
        for step in range(args.rolloutStorage_size):
            i_step += 1
            
            action = actor_critic.select_action(Variable(current_state))

            next_state, reward, done, _ = envs.step(action.squeeze(1).cpu().data.numpy())
            reward = torch.FloatTensor(reward).unsqueeze(1)
            masks = torch.FloatTensor(1-done).unsqueeze(1)
            if args.USE_CUDA == "True":
                masks = masks.cuda()
            #next_state = next_state.reshape(num_envs, *state_shape)
            next_state = hwc2chw(next_state)
            current_state = torch.FloatTensor(np.float32(next_state))
            rollout.insert(step, current_state, action.data, reward, masks)

        _, next_value = actor_critic(Variable(rollout.states[-1]))
        next_value = next_value.data
        # update the model for num_steps step, which is also the same step of the length of the rollout stotage
        ################# compute loss #################
        returns = rollout.compute_returns(next_value, args.gamma)
        logit, action_log_probs, values, entropy = actor_critic.evaluate_actions(Variable(rollout.states[:-1]).view(-1, *state_shape), Variable(rollout.actions).view(-1, 1))
        values = values.view(args.rolloutStorage_size, num_envs, 1)
        action_log_probs = action_log_probs.view(args.rolloutStorage_size, num_envs, 1)
        #advantage is the difference between predicted value and ground value
        advantages = Variable(returns) - values

        value_loss = advantages.pow(2).mean()
        action_loss = -(Variable(advantages.data) * action_log_probs).mean()
        optimizer.zero_grad()
        loss = value_loss * args.value_loss_coef + action_loss - entropy * args.entropy_coef
        loss.backward()
        nn.utils.clip_grad_norm_(actor_critic.parameters(), args.max_grad_norm)
        optimizer.step()
        rollout.after_update()
        
        if i_step % args.eval_freq == 0:
            print('i_step: {}'.format(i_step))
            if data_path != None:
                solved_rate, eval_avg_reward = test_the_agent(actor_critic, env_name, data_path, args.USE_CUDA, args.eval_num)

                wandb_session.log({'solved_ratio': solved_rate}, step=i_step)
                wandb_session.log({'avg_reward': eval_avg_reward}, step=i_step)
                print('solved ratio: {}'.format(solved_rate))

    envs.close()
