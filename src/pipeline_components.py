"""
Kubeflow Pipeline Components for Boston Housing Price Prediction
This module contains reusable components for the ML pipeline.
"""

from kfp import dsl
from typing import NamedTuple


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas==2.1.4", "scikit-learn==1.3.2", "numpy==1.24.3"]
)
def data_extraction(data_path: str) -> str:
    """
    Extracts and loads the dataset from the specified path.
    
    Args:
        data_path: Path to the raw data CSV file
        
    Returns:
        output_path: Path to the extracted data file
        
    This component simulates data extraction. In production, you would use
    'dvc get' or 'dvc import' to fetch versioned data from remote storage.
    """
    import pandas as pd
    import os
    
    print(f"[DATA EXTRACTION] Loading data from: {data_path}")
    
    # Create output directory
    output_dir = "/tmp/data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/extracted_data.csv"
    
    # Load and save data
    df = pd.read_csv(data_path)
    df.to_csv(output_path, index=False)
    
    print(f"[DATA EXTRACTION] Data extracted successfully")
    print(f"[DATA EXTRACTION] Shape: {df.shape}")
    print(f"[DATA EXTRACTION] Columns: {df.columns.tolist()}")
    print(f"[DATA EXTRACTION] Output saved to: {output_path}")
    
    return output_path


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas==2.1.4", "scikit-learn==1.3.2", "numpy==1.24.3"]
)
def data_preprocessing(input_data_path: str) -> NamedTuple('Outputs', [
    ('train_data_path', str),
    ('test_data_path', str),
    ('feature_names', list)
]):
    """
    Preprocesses the data: handles missing values, scales features, and splits into train/test sets.
    
    Args:
        input_data_path: Path to the input CSV file
        
    Returns:
        train_data_path: Path to the processed training data
        test_data_path: Path to the processed test data
        feature_names: List of feature column names
        
    This component performs data cleaning, feature scaling using StandardScaler,
    and splits data into 80% training and 20% test sets.
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import os
    from collections import namedtuple
    
    print("[DATA PREPROCESSING] Starting data preprocessing...")
    print(f"[DATA PREPROCESSING] Loading data from: {input_data_path}")
    
    df = pd.read_csv(input_data_path)
    print(f"[DATA PREPROCESSING] Initial shape: {df.shape}")
    
    # Handle missing values
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    print(f"[DATA PREPROCESSING] Dropped {dropped_rows} rows with missing values")
    print(f"[DATA PREPROCESSING] Shape after cleaning: {df.shape}")
    
    # Separate features and target
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    
    feature_names = X.columns.tolist()
    print(f"[DATA PREPROCESSING] Number of features: {len(feature_names)}")
    print(f"[DATA PREPROCESSING] Features: {feature_names}")
    
    # Split the data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[DATA PREPROCESSING] Train set size: {len(X_train)}")
    print(f"[DATA PREPROCESSING] Test set size: {len(X_test)}")
    
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[DATA PREPROCESSING] Features scaled using StandardScaler")
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # Save processed data
    output_dir = "/tmp/processed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = f"{output_dir}/train_data.csv"
    test_path = f"{output_dir}/test_data.csv"
    
    # Combine features and target
    train_df = X_train_scaled.copy()
    train_df['PRICE'] = y_train.values
    test_df = X_test_scaled.copy()
    test_df['PRICE'] = y_test.values
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"[DATA PREPROCESSING] Training data saved to: {train_path}")
    print(f"[DATA PREPROCESSING] Test data saved to: {test_path}")
    print("[DATA PREPROCESSING] Preprocessing completed successfully")
    
    outputs = namedtuple('Outputs', ['train_data_path', 'test_data_path', 'feature_names'])
    return outputs(train_path, test_path, feature_names)


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas==2.1.4", "scikit-learn==1.3.2", "joblib==1.3.2"]
)
def model_training(train_data_path: str, n_estimators: int = 100, max_depth: int = 10) -> str:
    """
    Trains a Random Forest Regressor model on the training data.
    
    Args:
        train_data_path: Path to the training CSV file
        n_estimators: Number of trees in the forest (default: 100)
        max_depth: Maximum depth of the trees (default: 10)
        
    Returns:
        model_path: Path to the saved model file (.joblib format)
        
    This component trains a Random Forest model for house price prediction
    and saves the trained model artifact for later evaluation and deployment.
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    import os
    
    print("[MODEL TRAINING] Starting model training...")
    print(f"[MODEL TRAINING] Loading training data from: {train_data_path}")
    
    train_df = pd.read_csv(train_data_path)
    print(f"[MODEL TRAINING] Training data shape: {train_df.shape}")
    
    # Separate features and target
    X_train = train_df.drop('PRICE', axis=1)
    y_train = train_df['PRICE']
    
    print(f"[MODEL TRAINING] Features shape: {X_train.shape}")
    print(f"[MODEL TRAINING] Target shape: {y_train.shape}")
    
    # Train Random Forest model
    print(f"[MODEL TRAINING] Training Random Forest with n_estimators={n_estimators}, max_depth={max_depth}")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("[MODEL TRAINING] Model training completed")
    
    # Get feature importances
    feature_importance = dict(zip(X_train.columns, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print("[MODEL TRAINING] Top 5 important features:")
    for feat, importance in top_features:
        print(f"  - {feat}: {importance:.4f}")
    
    # Save model
    output_dir = "/tmp/model"
    os.makedirs(output_dir, exist_ok=True)
    model_path = f"{output_dir}/model.joblib"
    
    joblib.dump(model, model_path)
    print(f"[MODEL TRAINING] Model saved to: {model_path}")
    print("[MODEL TRAINING] Training process completed successfully")
    
    return model_path


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas==2.1.4", "scikit-learn==1.3.2", "joblib==1.3.2"]
)
def model_evaluation(model_path: str, test_data_path: str) -> NamedTuple('Outputs', [
    ('accuracy', float),
    ('rmse', float),
    ('r2_score', float),
    ('mae', float),
    ('metrics_path', str)
]):
    """
    Evaluates the trained model on test data and computes performance metrics.
    
    Args:
        model_path: Path to the saved model file
        test_data_path: Path to the test CSV file
        
    Returns:
        accuracy: Percentage of predictions within 20% of actual value
        rmse: Root Mean Squared Error
        r2_score: R-squared score (coefficient of determination)
        mae: Mean Absolute Error
        metrics_path: Path to the JSON file containing all metrics
        
    This component loads the trained model, makes predictions on test data,
    and calculates various regression metrics to assess model performance.
    """
    import pandas as pd
    import joblib
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import json
    import os
    from collections import namedtuple
    import numpy as np
    
    print("[MODEL EVALUATION] Starting model evaluation...")
    print(f"[MODEL EVALUATION] Loading model from: {model_path}")
    model = joblib.load(model_path)
    
    print(f"[MODEL EVALUATION] Loading test data from: {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    print(f"[MODEL EVALUATION] Test data shape: {test_df.shape}")
    
    # Separate features and target
    X_test = test_df.drop('PRICE', axis=1)
    y_test = test_df['PRICE']
    
    print("[MODEL EVALUATION] Making predictions on test data...")
    y_pred = model.predict(X_test)
    
    # Calculate regression metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate custom "accuracy" metric for regression
    # Percentage of predictions within 20% of actual value
    accuracy = np.mean(np.abs((y_test - y_pred) / y_test) < 0.2) * 100
    
    # Calculate additional statistics
    mean_price = y_test.mean()
    std_price = y_test.std()
    
    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2_score': float(r2),
        'accuracy_within_20_percent': float(accuracy),
        'mean_actual_price': float(mean_price),
        'std_actual_price': float(std_price),
        'num_test_samples': len(y_test)
    }
    
    print("\n" + "="*50)
    print("[MODEL EVALUATION] EVALUATION RESULTS")
    print("="*50)
    print(f"  Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"  Mean Absolute Error (MAE):      ${mae:,.2f}")
    print(f"  R² Score:                        {r2:.4f}")
    print(f"  Accuracy (within 20%):           {accuracy:.2f}%")
    print(f"  Number of test samples:          {len(y_test)}")
    print(f"  Mean actual price:               ${mean_price:,.2f}")
    print("="*50 + "\n")
    
    # Save metrics to JSON file
    output_dir = "/tmp/metrics"
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = f"{output_dir}/metrics.json"
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[MODEL EVALUATION] Metrics saved to: {metrics_path}")
    print("[MODEL EVALUATION] Evaluation completed successfully")
    
    outputs = namedtuple('Outputs', ['accuracy', 'rmse', 'r2_score', 'mae', 'metrics_path'])
    return outputs(accuracy, rmse, r2, mae, metrics_path)