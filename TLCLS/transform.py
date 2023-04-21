import os

def transform_map(src_folder, file_name):
    file = open(os.path.join(src_folder, file_name), 'r')
    lines = file.readlines()

    res_map = []

    for line in lines:
        res_line = ''

        for char in line:
            if line == '\n':
                break

            if char == '0':
                res_line += ' '

            if char == '1':
                res_line += '#'

            if char == '2':
                res_line += '@'

            if char == '3':
                res_line += '$'

            if char == '4':
                res_line += '.'

            if char == '\n':
                res_line += '\n'

        res_map.append(res_line)

    # print(res_map)

    # write to file with name file_name and leading 0s
    new_file_name = file_name.split('.')[0].zfill(3) + '.txt'

    with open(f'maps/8x8/pcgrl/{new_file_name}', 'w') as f:
        for line in res_map:
            f.write(''.join(line))


folder = '../gym-pcgrl/shared_runs/8x8/sokoban/sokoban_turtle_2_2_log/generated/'

if __name__ == '__main__':
    for filename in os.listdir(folder):
        print(filename)
        if filename.endswith(".txt"):
            transform_map(folder, filename)
