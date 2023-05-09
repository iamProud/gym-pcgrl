import os

def transform_map(file, target_dir, iteration=0):
    file = open(file, 'r')
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

    # write to file with name file_name and leading 0s
    new_file_name = str(iteration) + file.split('/')[-1].split('.')[0].zfill(3) + '.txt'

    with open(os.path.join(target_dir, new_file_name), 'w') as f:
        for line in res_map:
            f.write(''.join(line))


folder = '../runs/sokoban_solver_turtle_1_0_log/'

if __name__ == '__main__':
    for filename in os.listdir(folder+ 'generated'):
        print(filename)
        if filename.endswith(".txt"):
            transform_map(folder, filename)
