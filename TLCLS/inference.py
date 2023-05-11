import os
import torch
from TLCLS.common.ActorCritic import ActorCritic
from TLCLS.common.test_the_agent import test_the_agent
import csv

def get_solver_agent(model_path, device='cpu'):
    state_shape = (3, 80, 80)
    num_actions = 5
    model = ActorCritic(state_shape, num_actions=num_actions)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    if 'cuda' in device:
        model.cuda()
    return model

#####
# Main
#####
save = True
save_file = 'pcg-solver_10_10-result.csv'
# run_name = 'run-20230329_132018-j5v9pcpn'

if __name__ == '__main__':
    model_path = '../shared_runs/5x5/sokoban/sokoban_solver_turtle_10_10_log/solver_model/model.pkl'
    model = get_solver_agent(model_path)

    env_name = 'Curriculum-Sokoban-v2'

    folder = '../shared_runs/5x5/sokoban/sokoban_solver_turtle_10_7_log/transformed'
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith('.txt'):
            continue
        avg_solved, reward_mean = test_the_agent(agent=model, env_name=env_name, data_path=os.path.join(folder, filename), USE_CUDA=False, eval_num=10, display=False)
        print('filename: ', filename)
        print('avg_solved: ', avg_solved)
        print('reward_mean: ', reward_mean, '\n')

        # save to .csv file
        if save:
            csv_file = os.path.join(folder, save_file)
            with open(csv_file, 'a') as f:
                # create the csv writer
                writer = csv.writer(f)

                # write a row to the csv file
                writer.writerow([filename, avg_solved, reward_mean])

    # avg_solved, reward_mean = test_the_agent(agent=model, env_name=env_name, data_path=os.path.join(folder, '048.txt'),
    #                                          USE_CUDA=False, eval_num=10, display=False)
    # print('avg_solved: ', avg_solved)
    # print('reward_mean: ', reward_mean, '\n')
