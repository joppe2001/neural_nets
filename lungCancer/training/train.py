from lungCancer.lungModel import LungModel
from trainer import Trainer
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lungCancer.preprocessing.preprocess_functions import create_dataloaders

def main():
    # Load preprocessed data using joblib
    processed_dict = joblib.load('../preprocessed_data.joblib')

    # Extract data from dictionary
    X_train_processed = processed_dict['X_train']
    X_val_processed = processed_dict['X_val']
    X_test_processed = processed_dict['X_test']
    y_train = processed_dict['y_train']
    y_val = processed_dict['y_val']
    y_test = processed_dict['y_test']
    preprocessor = processed_dict['preprocessor']

    # Create DataLoaders using your CancerDataset class
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train_processed, X_val_processed, X_test_processed,
        y_train, y_val, y_test,
        batch_size=32  # Adjust batch size as needed
    )

    # Initialize model
    input_features = X_train_processed.shape[1]  # Number of features after preprocessing
    model = LungModel(features_in=input_features)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=0.001,
        weight_decay=1e-5
    )

    # Train model
    history = trainer.train(num_epochs=50, early_stopping_patience=10)


if __name__ == "__main__":
    main()
