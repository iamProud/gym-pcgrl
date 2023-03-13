import json

game = 'sokoban'
representation = 'turtle'
run_idx = 1
render = False
is_inference = False

config = dict(
    width = 8,
    height = 8,
    change_percentage = 0.2,
    solver_power = 20000
)

game_path = f'shared_runs/{config["width"]}x{config["height"]}/{game}'
run_path = f'{game_path}/{game}_{representation}_{run_idx}_log/'

config['cropped_size'] = 2 * max(config['width'], config['height'])

with open(game_path+"/config.json", "r") as f:
    config_file = json.load(f)
    run_config = config_file['run'][str(run_idx)]

    for key, value in run_config.items():
         config[key] = config_file[key][value]

print(f'Running {game} with {representation} representation - run {run_idx}')
