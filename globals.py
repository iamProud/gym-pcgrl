import json

game = 'pushing'
representation = 'turtle'
run_idx = 1
render = True
is_inference = True

with open("shared_runs/config.json", "r") as f:
    config = json.load(f)
    run_config = config['run'][str(run_idx)]
    if run_config:
        global_probabilities = config['probabilities'][run_config['probabilities']]
        global_target_solution = config['target_solution'][run_config['target_solution']]
        global_max_crates = config['max_crates'][run_config['max_crates']]


print(f'Running {game} with {representation} representation - run {run_idx}')
