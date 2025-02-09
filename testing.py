import torch
import torch.nn as nn

from rich_logger import RichLogger

logger = RichLogger()

class AlzeimerModel(nn.Module):
    def __init__(self, features_in, h1=256, h2=128, h3=64, h4=32, features_out=1):
        super().__init__()

        self.layers = nn.Sequential(
            # Input layer with larger initial hidden size
            nn.Linear(features_in, h1),
            nn.BatchNorm1d(h1),
            nn.LeakyReLU(0.1),  # LeakyReLU instead of ReLU
            nn.Dropout(0.4),

            # Additional residual-style layer
            nn.Linear(h1, h1),
            nn.BatchNorm1d(h1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),

            # Gradual reduction in layer size
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(h3, h4),
            nn.BatchNorm1d(h4),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            # Output layer
            nn.Linear(h4, features_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class ModelCheckpoint:
    def __init__(self, path='best_model.pth'):
        self.best_accuracy = 0
        self.path = path

    def __call__(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            torch.save(model.state_dict(), self.path)
            return True
        return False


# Initialize checkpoint
checkpoint = ModelCheckpoint()


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Read the data
df = pd.read_csv("alzheimers.csv")

# Define numerical and categorical columns
numerical_columns = ['Age', 'BMI', 'Cognitive Test Score', 'Air Pollution Exposure']
categorical_columns = [
    'Gender', 'Education Level', 'Physical Activity Level',
    'Smoking Status', 'Alcohol Consumption', 'Diabetes',
    'Hypertension', 'Cholesterol Level', 'Family_history',
    'Depression Level', 'Sleep Quality', 'Dietary Habits',
    'Employment Status', 'Marital Status', 'Genetic Risk Factor (APOE-ε4 allele)',
    'Social Engagement Level', 'Income Level', 'Stress Levels',
    'Urban vs Rural Living'
]

# Convert 'Air Pollution Exposure' from categorical (High/Medium/Low) to numeric
pollution_map = {'Low': 0, 'Medium': 0.5, 'High': 1}
df['Air Pollution Exposure'] = df['Air Pollution Exposure'].map(pollution_map)

# Now scale the numerical columns
scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=categorical_columns)

# Convert Alzheimer's Diagnosis to binary
df_encoded['Alzeimer_diagnosis'] = (df_encoded['Alzeimer_diagnosis'] == 'Yes').astype(int)

# Prepare features and target
X = df_encoded.drop(['Country', 'Alzeimer_diagnosis'], axis=1).astype(float)
y = df_encoded['Alzeimer_diagnosis'].astype(float)

# Print shape to verify
print("Shape of features:", X.shape)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alzeimerModel = AlzeimerModel(X.shape[1])


# Convert data to PyTorch tensors
x_train = torch.FloatTensor(x_train.values)
x_test = torch.FloatTensor(x_test.values)
y_train = torch.FloatTensor(y_train.values).reshape(-1, 1)
y_test = torch.FloatTensor(y_test.values).reshape(-1, 1)

criterion = nn.BCELoss()
# Initialize with smaller learning rate
optimizer = torch.optim.Adam(alzeimerModel.parameters(), lr=0.001)

# More aggressive learning rate reduction
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.2,  # More aggressive reduction
    patience=100,  # Reduced patience
    min_lr=1e-6,  # Add minimum learning rate
    verbose=True
)

best_accuracy = 0
no_improve_count = 0

epochs = 1000
for i in range(epochs):
    # Training
    alzeimerModel.train()
    y_pred = alzeimerModel(x_train)
    train_loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Validation
    alzeimerModel.eval()
    with torch.no_grad():
        y_pred_test = alzeimerModel(x_test)
        test_loss = criterion(y_pred_test, y_test)
        predicted = (y_pred_test >= 0.5).float()
        accuracy = (predicted == y_test).float().mean()

    # Save best model
    if checkpoint(alzeimerModel, accuracy):
        no_improve_count = 0
    else:
        no_improve_count += 1

    # Early stopping
    if no_improve_count >= 100:  # Stop if no improvement for 500 epochs
        print(f"Early stopping! Best accuracy: {checkpoint.best_accuracy:.2f}%")
        break

    scheduler.step(test_loss)

    if i % 30 == 0:
        metrics = {
            "Epoch": i,
            "Training Loss": f"{train_loss.item():.4f}",
            "Validation Loss": f"{test_loss.item():.4f}",
            "Accuracy": f"{accuracy.item() * 100:.2f}%",
            "Best Accuracy": f"{checkpoint.best_accuracy * 100:.2f}%",
            "Learning Rate": f"{optimizer.param_groups[0]['lr']:.6f}"
        }
        logger.print_metrics(metrics, title=f"Training Progress - Epoch {i}")

# Load best model at the end
alzeimerModel.load_state_dict(torch.load('best_model.pth'))

