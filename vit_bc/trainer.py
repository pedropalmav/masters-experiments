from tqdm import tqdm
import torch

class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        self.model.to(self.device)

    def _compute_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        inputs = inputs.float()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        correct = self.accuracy_score(targets, outputs)
        return loss, correct

    def train_step(self, dataloader):
        self.model.train()
        total_loss = 0
        total_items = len(dataloader)
        correct = 0

        for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), total=total_items):
            loss, batch_correct = self._compute_batch(inputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += batch_correct

        return total_loss / total_items, 100 * correct / total_items

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_items = len(dataloader)
        correct = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                loss, batch_correct = self._compute_batch(inputs, targets)

                total_loss += loss.item()
                correct += batch_correct

        return total_loss / total_items, 100 * correct / total_items

    def fit(self, train_dataloader, validation_dataloader, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_step(train_dataloader)
            val_loss, val_acc = self.validate(validation_dataloader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    def accuracy_score(self, y_true, outputs):
        _, predicted = torch.max(outputs, 1)
        return (predicted == y_true).sum().item()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)