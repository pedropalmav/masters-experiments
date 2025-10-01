import torch
import torch.utils.data as data
import numpy as np
from vit import ViTBC
from BCDataset import BCDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = torch.load("data/sokoban_manual_data.pt", weights_only=False)
print(f"Dataset size: {len(dataset)}")
dataloader = data.DataLoader(dataset, batch_size=6, shuffle=True, num_workers=1)

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
num_epochs = 200
vit.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.permute(0, 3, 1, 2)  # Change shape from (B, H, W, C) to (B, C, H, W)
        inputs = inputs.float()
        optimizer.zero_grad()
        outputs = vit(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

torch.save(vit.state_dict(), "vit_sokoban.pth")