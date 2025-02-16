import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from .constants import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, TARGET_COLUMN

def clean_data(df):
    # Convert boolean strings to actual booleans
    bool_columns = ['Second_Hand_Smoke', 'Insurance_Coverage', 'Screening_Availability',
                    'Clinical_Trial_Access', 'Language_Barrier', 'Family_History',
                    'Indoor_Smoke_Exposure', 'Tobacco_Marketing_Exposure']

    for col in bool_columns:
        df = df.with_columns(
            pl.when(pl.col(col) == 'Yes').then(1)
              .when(pl.col(col) == 'No').then(0)
              .alias(col)
        )

    # Handle missing values
    df = df.with_columns([
        pl.col('Age').fill_null(pl.col('Age').median()),
        *[pl.col(col).fill_null('Unknown') for col in CATEGORICAL_COLUMNS]
    ])

    return df

def engineer_features(df):
    # Create age groups
    df = df.with_columns([
        pl.when(pl.col('Age').is_between(30, 45)).then('Young_Adult')
          .when(pl.col('Age').is_between(46, 60)).then('Middle_Age')
          .when(pl.col('Age').is_between(61, 75)).then('Senior')
          .when(pl.col('Age').is_between(76, 90)).then('Elderly')
          .otherwise('Unknown')
          .alias('Age_Group'),

        # Risk score combination
        (pl.col('Mortality_Risk') * (1 - pl.col('5_Year_Survival_Probability')))
          .alias('Combined_Risk'),

        # Convert Air_Pollution_Exposure to numeric
        pl.when(pl.col('Air_Pollution_Exposure') == 'Low').then(0)
          .when(pl.col('Air_Pollution_Exposure') == 'Medium').then(1)
          .when(pl.col('Air_Pollution_Exposure') == 'High').then(2)
          .alias('Air_Pollution_Numeric')
    ])

    return df

def create_preprocessor():
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_COLUMNS),
            ('cat', categorical_transformer, CATEGORICAL_COLUMNS)
        ],
        remainder='drop'  # This will drop any other columns
    )

    return preprocessor

def preprocess_data(df):
    # Split data first
    X = df.drop(TARGET_COLUMN)
    y = df.select(TARGET_COLUMN).with_columns(
        pl.when(pl.col(TARGET_COLUMN) == 'Yes').then(1)
          .when(pl.col(TARGET_COLUMN) == 'No').then(0)
          .alias(TARGET_COLUMN)
    )
    print("Class distribution:", y.value_counts(normalize=True))

    # Convert to numpy for sklearn
    X = X.to_numpy()
    y = y.to_numpy().flatten()

    # Create train/val/test splits
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Fit preprocessor on training data only
    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Convert back to polars Series for y values
    y_train = pl.Series(y_train)
    y_val = pl.Series(y_val)
    y_test = pl.Series(y_test)

    return (X_train_processed, X_val_processed, X_test_processed,
            y_train, y_val, y_test, preprocessor)

class CancerDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.to_numpy())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=1024):
    # Calculate weights for sampling
    value_counts = y_train.value_counts()
    class_counts = {value: count for value, count in zip(value_counts.to_numpy(), value_counts.counts())}
    weights = torch.FloatTensor([1 / class_counts[label] for label in y_train.unique().to_numpy()])
    sampler = WeightedRandomSampler(weights, len(weights))

    # Create all datasets
    train_dataset = CancerDataset(X_train, y_train)
    val_dataset = CancerDataset(X_val, y_val)
    test_dataset = CancerDataset(X_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader