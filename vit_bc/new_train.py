import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from vit import ViTBC
from BCDataset import BCDataset
from trainer import Trainer

def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)

dataset_path = os.path.join(os.path.dirname(__file__), "data", "10000_levels")
train_dataset = torch.load(os.path.join(dataset_path, "train_trajectories_250m.pt"), weights_only=False)
validation_dataset = torch.load(os.path.join(dataset_path, "valid_trajectories_250m.pt"), weights_only=False)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=1)
validation_dataloader = DataLoader(validation_dataset, batch_size=8, shuffle=False, num_workers=1)

vit = ViTBC(
        image_size=8,
        patch_size=1,
        num_layers=3,
        num_heads=8,
        hidden_dim=64,
        mlp_dim=128,
        image_channels=7,
        num_classes=5,
    )

num_epochs = 3
optimizer = torch.optim.Adam(vit.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

trainer = Trainer(vit, optimizer, criterion)

trainer.fit(train_dataloader, validation_dataloader, num_epochs)

torch.save(vit.state_dict(), os.path.join(os.path.dirname(__file__), "models", "vit_bc_refactor.pth"))

plot_loss_curve(trainer.history["train_loss"], trainer.history["val_loss"], os.path.join(os.path.dirname(__file__), "..", "imgs", "refactor.png"))