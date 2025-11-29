"""
Standalone Model Training Script for Boston Housing Price Prediction
This script can be run independently for testing purposes.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import numpy as np
import os


def load_data(data_path):
    """Load the dataset from CSV file."""
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def preprocess_data(df, test_size=0.2, random_state=42):
    """Preprocess the data: clean, scale, and split."""
    print("\nPreprocessing data...")
    
    # Handle missing values
    df = df.dropna()
    print(f"Shape after removing NaN: {df.shape}")
    
    # Separate features and target
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()


def train_model(X_train, y_train, n_estimators=100, max_depth=10):
    """Train a Random Forest Regressor model."""
    print("\nTraining Random Forest model...")
    print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model training completed!")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on test data."""
    print("\nEvaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = np.mean(np.abs((y_test - y_pred) / y_test) < 0.2) * 100
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"Mean Absolute Error (MAE):      ${mae:,.2f}")
    print(f"R² Score:                        {r2:.4f}")
    print(f"Accuracy (within 20%):           {accuracy:.2f}%")
    print("="*50)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'accuracy': accuracy
    }


def save_model(model, output_path='model/trained_model.joblib'):
    """Save the trained model to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"\nModel saved to: {output_path}")


def main():
    """Main function to orchestrate the training pipeline."""
    # Configuration
    DATA_PATH = 'data/raw/raw_data.csv'
    MODEL_PATH = 'model/trained_model.joblib'
    
    # Load data
    df = load_data(DATA_PATH)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, MODEL_PATH)
    
    print("\n✓ Training pipeline completed successfully!")


if __name__ == "__main__":
    main()