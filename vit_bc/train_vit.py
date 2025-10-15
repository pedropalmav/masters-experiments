import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from vit import ViTBC
from BCDataset import BCDataset

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        # dropout=0.1
    )

vit.to(device)

num_epochs = 25
optimizer = torch.optim.Adam(vit.parameters(), lr=1e-3) # weight_decay=1e-4)
# scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=5e-6)

criterion = torch.nn.CrossEntropyLoss()
best_val_loss = float("inf")
patience = 25
patience_counter = 0

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    vit.train()
    total_train_loss = 0.0
    total_train = 0
    correct_train = 0

    for batch_idx, (inputs, targets) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.float()
        optimizer.zero_grad()
        outputs = vit(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total_train += targets.size(0)
        correct_train += (predicted == targets).sum().item()

    vit.eval()
    total_val_loss = 0.0
    total_val = 0
    correct_val = 0
    
    with torch.no_grad():
        for inputs, targets in validation_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.float()
            outputs = vit(inputs)
            loss = criterion(outputs, targets)
            total_val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_val += targets.size(0)
            correct_val += (predicted == targets).sum().item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_val_loss = total_val_loss / len(validation_dataloader)
    train_accuracy = 100 * correct_train / total_train
    val_accuracy = 100 * correct_val / total_val

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%")

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    # scheduler.step()

    # if avg_val_loss < best_val_loss:
    #     best_val_loss = avg_val_loss
    #     patience_counter = 0
    #     torch.save(vit.state_dict(), os.path.join(os.path.dirname(__file__), "models", "vit_bc.pth"))
    # else:
    #     patience_counter += 1
    
    # if patience_counter >= patience:
    #     print("Early stopping triggered")
    #     break


torch.save(vit.state_dict(), os.path.join(os.path.dirname(__file__), "models", "vit_bc.pth"))

plot_loss_curve(train_losses, val_losses, os.path.join(os.path.dirname(__file__), "..", "imgs", "loss_curve.png"))