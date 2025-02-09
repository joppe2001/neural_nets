import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, in_features=4, h1=2, h2=8, out_features=1):
        super().__init__()

        # hidden layer 1
        self.fc1 = nn.Linear(in_features, h1)
        self.sigmoid1 = nn.Sigmoid()

        # hidden layer 2
        self.fc2 = nn.Linear(h1, h2)
        self.sigmoid2 = nn.Sigmoid()

        # output layer
        self.fc3 = nn.Linear(h2, out_features)
        self.sigmoid3 = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid1(self.fc1(x))
        x = self.sigmoid2(self.fc2(x))
        x = self.sigmoid3(self.fc3(x))
        return x


# Instantiate the model
model = SimpleModel()

# define the training data
x_train = torch.FloatTensor([
    [1, 1, 0, 1],
    [0, 0, 0, 1],
    [1, 1, 1, 0],
    [1, 0, 1, 1],
    [0, 1, 1, 0],
    [1, 0, 0, 0],
    [1, 1, 1, 1],
    [0, 1, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 0, 1]
])

y_train = torch.FloatTensor([[1, 0, 1, 1, 0, 1, 1, 0, 1, 0]]).T

# define the validation data
x_val = torch.FloatTensor([
    [1, 1, 0, 1],
    [0, 1, 0, 1],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 1]
])

y_val = torch.FloatTensor([[1, 0, 1, 0, 0]]).T


criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Training loop
epochs = 10000
for epoch in range(epochs):
    # Forward pass
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights

    # Print progress
    if epoch % 1000 == 0:
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Don't compute gradients for validation
            val_pred = model(x_val)
            print(f"Epoch {epoch}, Validation Predictions:")
            print(val_pred.numpy())
        model.train()  # Set model back to training mode\

# Final evaluation
model.eval()
with torch.no_grad():
    final_predictions = model(x_val)
    print("\nFinal Validation Predictions:")
    print(final_predictions.numpy())
    print("\nTrue Validation Values:")
    print(y_val.numpy())

# After your existing evaluation code
model.eval()
with torch.no_grad():
    final_predictions = model(x_val)

    # Add this part for cleaner output
    predictions_rounded = (final_predictions > 0.5).float()  # Threshold at 0.5
    correct = (predictions_rounded == y_val).sum().item()
    total = len(y_val)

    print("\nPredictions vs True Values:")
    for pred, true in zip(final_predictions.numpy(), y_val.numpy()):
        print(f"Predicted: {pred[0]:.8f}, True: {true[0]}")

    print(f"\nAccuracy: {correct}/{total} ({100 * correct / total:.2f}%)")