import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt


class Trainer:
    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            test_loader,
            learning_rate=0.001,
            weight_decay=1e-5,
            device='mps'  # For M1/M2/M3 Mac
    ):
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.1,
            patience=5,
            verbose=True
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []

        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target.unsqueeze(1))

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Collect metrics
            total_loss += loss.item()
            predictions.extend(output.detach().cpu().numpy())
            true_labels.extend(target.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        # Calculate epoch metrics
        epoch_loss = total_loss / len(self.train_loader)
        predictions = np.array(predictions).reshape(-1)
        true_labels = np.array(true_labels)
        auc = roc_auc_score(true_labels, predictions)
        accuracy = accuracy_score(true_labels, predictions > 0.5)

        return epoch_loss, auc, accuracy

    @torch.no_grad()
    def evaluate(self, loader, phase='val'):
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []

        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target.unsqueeze(1))

            total_loss += loss.item()
            predictions.extend(output.cpu().numpy())
            true_labels.extend(target.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(loader)
        predictions = np.array(predictions).reshape(-1)
        true_labels = np.array(true_labels)
        auc = roc_auc_score(true_labels, predictions)
        accuracy = accuracy_score(true_labels, predictions > 0.5)

        return avg_loss, auc, accuracy

    def train(self, num_epochs=50, early_stopping_patience=10):
        best_val_auc = 0
        patience_counter = 0
        history = {
            'train_loss': [], 'train_auc': [], 'train_acc': [],
            'val_loss': [], 'val_auc': [], 'val_acc': []
        }

        # Initialize wandb
        wandb.init(project="lung-cancer-prediction", config={
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "architecture": "LungModel",
            "dataset": "Lung Cancer",
            "epochs": num_epochs,
        })

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Training phase
            train_loss, train_auc, train_acc = self.train_epoch()

            # Validation phase
            val_loss, val_auc, val_acc = self.evaluate(self.val_loader, 'val')

            # Update learning rate scheduler
            self.scheduler.step(val_auc)

            # Log metrics
            metrics = {
                "train_loss": train_loss, "train_auc": train_auc, "train_acc": train_acc,
                "val_loss": val_loss, "val_auc": val_auc, "val_acc": val_acc,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }
            wandb.log(metrics)

            # Update history
            for k, v in metrics.items():
                if k in history:
                    history[k].append(v)

            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")

            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered!")
                    break

        # Final evaluation on test set
        self.model.load_state_dict(torch.load('best_model.pth'))
        test_loss, test_auc, test_acc = self.evaluate(self.test_loader, 'test')
        print(f"\nTest Results - Loss: {test_loss:.4f}, AUC: {test_auc:.4f}, Acc: {test_acc:.4f}")

        # Plot training history
        self.plot_training_history(history)

        wandb.finish()
        return history

    def plot_training_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_title('Loss over epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot AUC
        ax2.plot(history['train_auc'], label='Train AUC')
        ax2.plot(history['val_auc'], label='Val AUC')
        ax2.set_title('AUC over epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

