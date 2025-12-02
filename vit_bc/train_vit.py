import os
import argparse
import time
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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


parser = argparse.ArgumentParser(description="Train ViT for Behavior Cloning")
parser.add_argument("--epochs", type=int, help="Number of training epochs")
parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--warmup_epochs", type=int, default=10, help="Number of warm-up epochs")
parser.add_argument("--start_factor", type=float, default=1e-3, help="Starting learning rate for warm-up")
parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
parser.add_argument("--filename", type=str, help="Filename to save the model and results")
parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size")
args = parser.parse_args()

filename = f"vit_{args.filename}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = os.path.join(os.path.dirname(__file__), "data", "full_npy")

validation_dataset = BCDataset(
    states_path=os.path.join(dataset_path, "valid_states.npy"),
    actions_path=os.path.join(dataset_path, "valid_actions.npy")
)
print("Validation dataset loaded.")

train_states_files = sorted(glob.glob(os.path.join(dataset_path, "train_states_*.npy")))
train_actions_files = sorted(glob.glob(os.path.join(dataset_path, "train_actions_*.npy")))
train_datasets = []
for states_file, actions_file in zip(train_states_files, train_actions_files):
    train_dataset = BCDataset(
        states_path=states_file,
        actions_path=actions_file
    )
    train_datasets.append(train_dataset)
train_dataset = ConcatDataset(train_datasets)
print("Training dataset loaded.")

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

vit = ViTBC(
        image_size=8,
        patch_size=1,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        mlp_dim=args.hidden_dim * 4,
        image_channels=7,
        num_classes=5,
        dropout=args.dropout
    )

vit.to(device)

num_epochs = args.epochs
optimizer = torch.optim.Adam(vit.parameters(), lr=args.lr)
if args.warmup_epochs > 0:
    decay_epochs = num_epochs - args.warmup_epochs
    warmup = LinearLR(optimizer, start_factor=args.start_factor, end_factor=1.0, total_iters=args.warmup_epochs)
    decay = CosineAnnealingLR(optimizer, T_max=decay_epochs, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[args.warmup_epochs])

criterion = torch.nn.CrossEntropyLoss()
best_val_loss = float("inf")
patience = 15
patience_counter = 0

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    vit.train()
    total_train_loss = 0.0
    total_train = 0
    correct_train = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataloader)):
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
    if args.warmup_epochs > 0:
        scheduler.step()

    if args.early_stopping:
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(vit.state_dict(), os.path.join(os.path.dirname(__file__), "models", "full", filename + ".pth"))
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

if patience_counter < patience:
    torch.save(vit.state_dict(), os.path.join(os.path.dirname(__file__), "models", "full", filename + ".pth"))

plot_loss_curve(train_losses, val_losses, os.path.join(os.path.dirname(__file__), "..", "imgs", "full", filename + ".png"))