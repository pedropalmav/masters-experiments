import argparse
import gym
import numpy as np
import gym_sokoban
import torch
import os
from tqdm import tqdm
import utils.observation_parser as observation_parser
from utils.sokoban_visualizer import SokobanVisualizer
from vit import ViTBC

def test_level(env: gym.Env, room_id: int, render: bool = True):
    obs = env.reset(room_id=room_id)

    if render:
        visualizer = SokobanVisualizer()
        visualizer.render(obs)

    with torch.no_grad():
        while True:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            action = vit(obs_tensor).argmax(dim=1).item()
            obs, reward, done, _ = env.step(action)

            if render:
                visualizer.render(obs)

            if done:
                break
        
    return success(obs)

def success(obs: np.ndarray) -> int:
    return int(not np.any(obs[:, :, 2]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_levels", type=int, help="Number of levels to test")
    parser.add_argument("--render", default=False, help="Render the environment")
    parser.add_argument("--env_split", type=str, default="", help="Environment split")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit = ViTBC(
        image_size=8,
        patch_size=1,
        num_layers=5,
        num_heads=8,
        hidden_dim=64,
        mlp_dim=128,
        image_channels=7,
        num_classes=5,
        # dropout=0.1
    )
    vit.load_state_dict(
        torch.load(
            os.path.join(
                os.path.dirname(__file__),
                "models",
                "100000_levels",
                "vit_5_layers_0.003.pth"
            ),
            map_location=device
        )
    )
    vit.to(device)
    vit.eval()

    middle = {
        '': '-',
        'train': '-',
        'valid': '-valid-',
        'test': '-test-'
    }
    env = gym.make(f"Sokoban{middle[args.env_split]}v0")

    total_success = 0
    for level in tqdm(range(args.num_levels), desc="Testing levels"):
        total_success += test_level(env, level, args.render)
    
    print(f"Success rate: {total_success / args.num_levels:.2%}")