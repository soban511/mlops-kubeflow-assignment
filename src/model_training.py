"""
Standalone model training script for local testing
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston
import joblib
import json
import os


def load_data():
    """Load the Boston Housing dataset"""
    print("Loading Boston Housing dataset...")
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['PRICE'] = boston.target
    return df


def preprocess_data(df, test_size=0.2, random_state=42):
    """Preprocess and split the data"""
    print("Preprocessing data...")
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train, n_estimators=100, max_depth=10):
    """Train the Random Forest model"""
    print(f"Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model"""
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'r2_score': r2,
        'accuracy': max(0, r2) * 100
    }
    
    print(f"\nModel Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    
    return metrics


def main():
    """Main training pipeline"""
    # Create output directory
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model and metrics
    joblib.dump(model, 'models/random_forest_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nModel and metrics saved successfully!")


if __name__ == "__main__":
    main()
