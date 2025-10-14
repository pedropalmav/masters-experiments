import gym
import gym_sokoban
import numpy as np
import pygame
from sokoban_visualizer import SokobanVisualizer

def handle_player_input():
    global quit_game
    keys = pygame.key.get_pressed()
    action = 0
    if keys[pygame.K_RIGHT]:
        action = 4
    if keys[pygame.K_UP]:
        action = 1
    if keys[pygame.K_LEFT]:
        action = 3
    if keys[pygame.K_DOWN]:
        action = 2
    if keys[pygame.K_d]:
        action = 4
    if keys[pygame.K_w]:
        action = 1
    if keys[pygame.K_a]:
        action = 3
    if keys[pygame.K_s]:
        action = 2
    if keys[pygame.K_ESCAPE]:
        quit_game = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_game = True

    return action


def get_action():
    key = input("Enter direction (w/a/s/d): ")
    match(key):
        case 'w':
            return 1
        case 's':
            return 2
        case 'a':
            return 3
        case 'd':
            return 4
        case _:
            return 0
        
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


env = gym.make("Sokoban-v0")
obs = env.reset(room_id=0)

visualizer = SokobanVisualizer()
visualizer.render(obs)

while True:
    action = handle_player_input()
    obs, reward, done, _ = env.step(action)
    visualizer.render(obs)
    if done:
        break
