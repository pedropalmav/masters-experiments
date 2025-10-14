import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from vit import ViTBC
from BCDataset import BCDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = os.path.join(os.path.dirname(__file__), "data", "train_trajectories_250m.pt")
dataset = torch.load(dataset_path, weights_only=False)
print(f"Dataset size: {len(dataset)}")
dataloader = DataLoader(dataset, batch_size=10, num_workers=1)

dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
validation_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - validation_size

train_indices = list(range(0, train_size))
validation_indices = list(range(train_size, train_size + validation_size))
test_indices = list(range(train_size + validation_size, dataset_size))

train_dataset = Subset(dataset, train_indices)
validation_dataset = Subset(dataset, validation_indices)
test_dataset = Subset(dataset, test_indices)

train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)
validation_dataloader = DataLoader(validation_dataset, batch_size=10, shuffle=False, num_workers=1)
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=1)

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

vit.to(device)

optimizer = torch.optim.Adam(vit.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 20
best_val_loss = float("inf")
patience = 5
patience_counter = 0


for epoch in range(num_epochs):
    vit.train()
    total_train_loss = 0.0
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.float()
        optimizer.zero_grad()
        outputs = vit(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    vit.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in validation_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.float()
            outputs = vit(inputs)
            loss = criterion(outputs, targets)
            total_val_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_val_loss = total_val_loss / len(validation_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(vit.state_dict(), os.path.join(os.path.dirname(__file__), "models", "vit_bc.pth"))
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print("Early stopping triggered")
        break


torch.save(vit.state_dict(), os.path.join(os.path.dirname(__file__), "models", "vit_bc.pth"))