import joblib
from preprocessing.preprocess_functions import clean_data, engineer_features, preprocess_data, create_dataloaders
import polars as pl
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rich_logger import RichLogger

def main():
    logger = RichLogger()

    logger.section("Loading Data")
    # Load data using polars
    df = pl.read_csv('data/lung_cancer_data.csv')
    logger.info(f"Loaded dataset with shape: {df.shape}")

    logger.section("Cleaning Data")
    # Clean data
    df = clean_data(df)
    logger.info(f"Shape after cleaning: {df.shape}")

    # Print sample of first 5 rows
    logger.print_array(df.head().select(df.columns).to_numpy(),
                      title="Sample Data After Cleaning",
                      border_style="blue")

    logger.section("Feature Engineering")
    # Engineer features
    df = engineer_features(df)
    logger.info(f"Shape after feature engineering: {df.shape}")

    # Print summary of features added
    feature_info = {
        "Number of features": df.shape[1],
        "Original features": ", ".join(df.columns[:25]), # First 25 columns
        "Engineered features": ", ".join(df.columns[25:]) # Columns after 25
    }
    logger.print_dict(feature_info, title="Feature Information", border_style="green")

    logger.section("Preprocessing")
    # Preprocess
    processed_data = preprocess_data(df)
    logger.info("Data preprocessing completed")

    logger.section("Creating Dataloaders")
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(*processed_data[:6])

    # Print split info
    split_info = {
        "Training samples": len(train_loader.dataset),
        "Validation samples": len(val_loader.dataset),
        "Test samples": len(test_loader.dataset)
    }
    logger.print_metrics(split_info, title="Dataset Splits", border_style="yellow")

    # Save processed data
    processed_dict = {
        'X_train': processed_data[0],
        'X_val': processed_data[1],
        'X_test': processed_data[2],
        'y_train': processed_data[3].to_numpy(), # Convert polars Series to numpy
        'y_val': processed_data[4].to_numpy(),
        'y_test': processed_data[5].to_numpy(),
        'preprocessor': processed_data[6]
    }
    joblib.dump(processed_dict, 'preprocessed_data.joblib')
    logger.info("Saved preprocessed data to preprocessed_data.joblib")

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    main()