from torch.utils.data import Dataset
import torch
import numpy as np

class BCDataset(Dataset):
    def __init__(self, states_path, actions_path):
        self.states = np.load(states_path, mmap_mode='r')
        self.actions = np.load(actions_path, mmap_mode='r')
        
        assert len(self.states) == len(self.actions)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        state = torch.tensor(self.states[index])
        action = torch.tensor(self.actions[index])
        
        return state, action