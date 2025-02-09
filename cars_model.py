import torch
import torch.nn as nn
from rich_logger import RichLogger

logger = RichLogger()


class CarsModel(nn.Module):
    def __init__(self, features_in=3, h1=4, h2=8, h3=16, features_out=1, dropout_rate=0.2):
        super().__init__()

        self.fc1 = nn.Linear(features_in, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)


        self.fc4 = nn.Linear(h3, features_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x = self.dropout3(self.relu3(self.bn3(self.fc3(x))))
        x = self.sigmoid(self.fc4(x))
        return x


carModel = CarsModel()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau


df = pd.read_csv("cars.csv")

df["Gender"] = (df["Gender"] == "Male").astype(int)

age_scaler = MinMaxScaler()
salary_scaler = MinMaxScaler()

df["Age"] = age_scaler.fit_transform(df["Age"].values.reshape(-1, 1))
df["AnnualSalary"] = salary_scaler.fit_transform(df["AnnualSalary"].values.reshape(-1, 1))

x = df[["Gender", "Age", "AnnualSalary"]]
y = df["Purchased"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
x_train = torch.FloatTensor(x_train.values)
x_test = torch.FloatTensor(x_test.values)
y_train = torch.FloatTensor(y_train.values).reshape(-1, 1)
y_test = torch.FloatTensor(y_test.values).reshape(-1, 1)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(carModel.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


epochs = 100000

# Training with rich logging
logger.section("Starting Training")

for i in range(epochs):
    # Training
    carModel.train()
    y_pred = carModel(x_train)
    train_loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Validation
    carModel.eval()
    with torch.no_grad():
        y_pred_test = carModel(x_test)
        test_loss = criterion(y_pred_test, y_test)
        predicted = (y_pred_test >= 0.5).float()
        accuracy = (predicted == y_test).float().mean()

    # Log progress every 100 epochs
    if i % 1000 == 0:
        metrics = {
            "Epoch": i,
            "Training Loss": f"{train_loss.item():.4f}",
            "Validation Loss": f"{test_loss.item():.4f}",
            "Accuracy": f"{accuracy.item() * 100:.2f}%"
        }
        logger.print_metrics(metrics, title=f"Training Progress - Epoch {i}")

# Manual Testing
logger.section("Manual Testing")


def test_model():
    while True:
        try:
            logger.info("Enter the following information (or 'q' to quit):")
            gender = input("Gender (1 for Male, 0 for Female): ")
            if gender.lower() == 'q':
                break

            age = input("Age (18-100): ")
            if age.lower() == 'q':
                break

            salary = input("Annual Salary (20000-200000): ")
            if salary.lower() == 'q':
                break

            # Scale the inputs using the same scalers
            scaled_age = age_scaler.transform([[float(age)]])[0][0]
            scaled_salary = salary_scaler.transform([[float(salary)]])[0][0]

            # Create input tensor
            input_data = torch.FloatTensor([[float(gender), scaled_age, scaled_salary]])

            # Make prediction
            carModel.eval()
            with torch.no_grad():
                prediction = carModel(input_data)
                probability = prediction.item()
                decision = "Will buy" if probability >= 0.5 else "Will not buy"

            # Log results
            input_info = {
                "Gender": "Male" if gender == "1" else "Female",
                "Age": age,
                "Annual Salary": f"${salary}"
            }

            prediction_info = {
                "Probability": f"{probability:.2%}",
                "Prediction": decision
            }

            logger.print_items([
                (input_info, "Input Information", "blue"),
                (prediction_info, "Model Prediction", "green")
            ], side_by_side=True)

        except ValueError as e:
            logger.error("Please enter valid numeric values!")
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")

        continue_test = input("\nTry another prediction? (y/n): ")
        if continue_test.lower() != 'y':
            break

    logger.info("Testing completed!")


# Run the manual testing
test_model()