from TLCLS.gym_sokoban.envs.room_utils import generate_room
from TLCLS.gym_sokoban.envs.render_utils import room_to_rgb

from PIL import Image
import os

def get_max_file_idx(folder_path):
    idx = -1

    for file in os.listdir(folder_path):
        try:
            if file.endswith(".txt"):
                file_idx = int(file.split('.')[0].split('sokoban')[1])
                idx = max(idx, file_idx)

        except ValueError:
            pass
    return idx

def trasform_room(room_state):
    room_state_str = []

    for row in room_state:
        row_str = ''

        for i in row:
            if i == 0:
                row_str += '#'

            if i == 1:
                row_str += ' '

            if i == 2:
                row_str += '.'

            if i == 4:
                row_str += '$'

            if i == 5:
                row_str += '@'

        room_state_str.append(row_str)

    return room_state_str


if __name__ == '__main__':
    max_files = 200
    show_img = False

    config = {
        'width': 10,
        'height': 10,
        'num_boxes': 1,
    }

    save_path = f'maps/{config["width"]-2}x{config["height"]-2}/{config["num_boxes"]}'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    while True:
        n = get_max_file_idx(save_path)
        if n >= max_files-1:
            break

        # generate a new room
        try:
            _, room_state, _ = generate_room(
                dim=(config['width'], config['height']),
                num_boxes=config['num_boxes']
            )
        except (RuntimeWarning, RuntimeError):
            print('Skipping room... (invalid)')
            continue

        # show room
        if show_img:
            room_rgb = room_to_rgb(room_state)
            room_img = Image.fromarray(room_rgb, 'RGB')
            room_img.show()
            # input('Press enter to continue...')

        # transform room
        room_state_str = trasform_room(room_state)

        # check if room is already in folder
        room_in_folder = False

        ld = os.listdir(save_path)
        for file in ld:
            with open(f'{save_path}/{file}', 'r') as f:
                file_state_str = f.read().replace('\n', '')
                room_state_str_concat = ''.join(room_state_str)

                if room_state_str_concat == file_state_str:
                    room_in_folder = True
                    f.close()
                    break

            f.close()

        if room_in_folder:
            print('Skipping room... (duplicate)')
            continue


        # save room to .txt file
        file_name = f'sokoban{n+1:03d}.txt'

        with open(f'{save_path}/{file_name}', 'w') as f:
            # write 2d numpy array to file
            for row in room_state_str:
                f.write(row + '\n')

        f.close()
        print(f'File {file_name} saved to {save_path}')


