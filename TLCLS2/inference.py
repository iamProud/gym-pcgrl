import os
import torch
from TLCLS.common.ActorCritic import ActorCritic
from TLCLS.common.test_the_agent import test_the_agent

def get_solver_agent(model_path):
    state_shape = (3, 80, 80)
    num_actions = 5
    model = ActorCritic(state_shape, num_actions=num_actions)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

#####
# Main
#####
# run_name = 'run-20230329_132018-j5v9pcpn'

if __name__ == '__main__':
    model_path = os.path.join('wandb', 'model.pkl')
    model = get_solver_agent(model_path)

    env_name = 'Curriculum-Sokoban-v2'

    folder = 'maps/8x8/pcgrl'
    for filename in sorted(os.listdir(folder)):
        avg_solved, reward_mean = test_the_agent(agent=model, env_name=env_name, data_path=os.path.join(folder, filename), USE_CUDA=False, eval_num=3, display=False)
        print('filename: ', filename)
        print('avg_solved: ', avg_solved)
        print('reward_mean: ', reward_mean, '\n')

    # avg_solved, reward_mean = test_the_agent(agent=model, env_name=env_name, data_path=os.path.join(folder, '048.txt'),
    #                                          USE_CUDA=False, eval_num=10, display=False)
    # print('avg_solved: ', avg_solved)
    # print('reward_mean: ', reward_mean, '\n')
