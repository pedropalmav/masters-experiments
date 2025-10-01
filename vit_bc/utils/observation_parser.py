import numpy as np

def onehot_decode(onehot):
    return np.where(onehot == 1)[0][0]

def position2char(pos):
    match(pos):
        case 0:
            return '#'
        case 1:
            return ' '
        case 2:
            return '$'
        case 3:
            return '!'
        case 4:
            return '@'
        case 5:
            return '%'
        case 6:
            return '.'

def format_obs(obs):
    for row in range(len(obs)):
        for col in range(len(obs[row])):
            index = onehot_decode(obs[row][col])
            print(position2char(index), end="")
        print()