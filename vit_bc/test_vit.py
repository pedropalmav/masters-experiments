import gym
import gym_sokoban
import torch
import os
import utils.observation_parser as observation_parser
from utils.sokoban_visualizer import SokobanVisualizer
from vit import ViTBC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit = ViTBC(
        image_size=8,
        patch_size=1,
        num_layers=4,
        num_heads=4,
        hidden_dim=128,
        mlp_dim=256,
        image_channels=7,
        num_classes=5
    )
vit.load_state_dict(
    torch.load(os.path.join(os.path.dirname(__file__), "models", "vit_one_trajectory.pth"), map_location=device)
)
vit.to(device)
vit.eval()

env = gym.make("Sokoban-v0")
obs = env.reset(room_id=0)

visualizer = SokobanVisualizer()
visualizer.render(obs)

with torch.no_grad():
    while True:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        action = vit(obs_tensor).argmax(dim=1).item()
        obs, reward, done, _ = env.step(action)

        visualizer.render(obs)

        if done:
            break