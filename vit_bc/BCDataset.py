from torch.utils.data import Dataset

class BCDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        state = self.states[index]
        action = self.actions[index]
        return state, action