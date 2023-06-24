'''
Transforms the map from integer representation to Sokoban representation
'''
def transform_map(lines):
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

        if res_line == '':
            continue

        res_map.append(res_line)

    return res_map
