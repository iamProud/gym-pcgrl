import json

game = 'sokoban_tlcls'
representation = 'turtle'
run_idx = 2
render = False
is_inference = False
device='auto'

config = dict(
    width = 8,
    height = 8,
    change_percentage = 0.2,
    solver_power = 20000
)

game_path = f'shared_runs/{config["width"]}x{config["height"]}/sokoban'
run_path = f'{game_path}/sokoban_{representation}_{run_idx}_2_log/'

config['cropped_size'] = 2 * max(config['width'], config['height'])

print(f'Running {game} with {representation} representation - run {run_idx}')

config['max_crates'] = 2
config['target_solution'] = 15
config['probabilities'] = {
    "empty": 0.6,
    "solid": 0.34,
    "player": 0.02,
    "crate": 0.02,
    "target": 0.02
}



# with open(game_path+"/config.json", "r") as f:
#     config_file = json.load(f)
#     run_config = config_file['run'][str(run_idx)]
#
#     for key, value in run_config.items():
#         config[key] = config_file[key][value]
#         print(f'### {key} = {config[key]}')
