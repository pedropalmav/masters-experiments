import gym
import gym_sokoban
import torch
import os
import utils.observation_parser as observation_parser
from BCDataset import BCDataset

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

env = gym.make("Sokoban-v0")
obs = env.reset(room_id=0)
observation_parser.format_obs(obs)

states = [obs]
actions = []

while True:
    action = get_action()
    actions.append(action)

    obs, reward, done, _ = env.step(action)
    states.append(obs)
    observation_parser.format_obs(obs)
    if done:
        break

actions.append(0)  # No-op action at the end

dataset = BCDataset(states, actions)
script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, "..", "data", "sokoban_manual_data.pt")
torch.save(dataset, path)
print(f"Dataset saved to {path}")