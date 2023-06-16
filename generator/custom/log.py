import os
import json
import numpy as np

from gym_pcgrl.envs.helper import safe_map


"""
Logs the results of the inference to a file
param info: the info dictionary
param path_generated: the path to the generated level
return: True if the inference was successful, False otherwise
"""
def log_inference(info, **kwargs):
    path_generated = kwargs.get('path_generated')
    success_condition = True
    log_json = kwargs.get('log_json', False)

    if kwargs.get('infer') is not None:
        crates_condition = True if info.get('crate') is None or kwargs['infer'].get('crate') is None \
            else kwargs['infer']['crate'].get('min', 1)  <= info.get('crate') <= kwargs['infer']['crate'].get('max', np.inf)

        solver_condition = True if info.get('solver') is None or kwargs['infer'].get('solver') is None \
            else kwargs['infer']['solver'].get('min', 1) <= info.get('solver') <= kwargs['infer']['solver'].get('max', np.inf)

        solution_condition = True if info.get('sol-length') is None or kwargs['infer'].get('sol-length') is None \
            else kwargs['infer']['sol-length'].get('min', 1) <= info.get('sol-length') <= kwargs['infer']['sol-length'].get('max', np.inf)

        success_condition = solution_condition and crates_condition and solver_condition

    # write general info to file
    if log_json:
        with open(path_generated + "/info.json", "r") as f:
            data = json.load(f)
            data["trials"] += 1
            successful = data['success-rate'] * (data['trials'] - 1)
            data['success-rate'] = (successful + 1) / (data['trials']) if success_condition else successful / (data['trials'])
        with open(path_generated + "/info.json", "w") as f:
            json.dump(data, f)

    if success_condition:
        log_successful(info, path_generated, log_json)
    else:
        if log_json:
            log_failed(info, path_generated)

    return success_condition


"""
Logs the results of the successful inference to a file
"""


def log_successful(info, path_generated, log_json):
    if log_json:
        with open(path_generated + "/info.json", "r") as f:
            data = json.load(f)
            successful = round(data['success-rate'] * (data['trials'] - 1))

            data['avg-sol-length'] = (data['avg-sol-length'] * successful + info["sol-length"]) / (successful + 1)
            data['avg-crates'] = (data['avg-crates'] * successful + info["crate"]) / (successful + 1)
            free_ratio = np.count_nonzero(info['map'] == 0) / ((info['map'].shape[0]-2) * (info['map'].shape[1]-2))
            data['avg-free-percent'] = (data['avg-free-percent'] * successful + free_ratio) / (successful + 1)
        with open(path_generated + "/info.json", "w") as f:
            json.dump(data, f)

    # get file number
    listdir = os.listdir(path_generated)

    if len(listdir) == 0:
        file_count = 0
    else:
        file_count = -1
        for generated_file in listdir:
            try:
                val = int(generated_file.split('.')[0])
                file_count = max(file_count, val)
            except ValueError:
                continue

        file_count += 1

    # save map as image
    info['img'].save(f'{path_generated}/{file_count}.jpeg')

    # save map as .txt
    safe_map(info['map'], info['solution'], path_generated, file_count)


"""
Logs the results of the failed inference to a file
failed obj: 'total', '0-player', '2+players', 'region', 'crate-target'
"""


def log_failed(info, path_generated):
    with open(path_generated + "/info.json", "r") as f:
        data = json.load(f)
        data['failed']['total'] += 1
        data['failed']['0-player'] += 1 if info['player'] == 0 else 0
        data['failed']['2+players'] += 1 if info['player'] >= 2 else 0
        data['failed']['region'] += 1 if info['regions'] > 1 else 0
        data['failed']['crate-target'] += 1 if info['crate'] != info['target'] else 0
    with open(path_generated + "/info.json", "w") as f:
        json.dump(data, f)
