"""
Kubeflow Pipeline Components for MLOps Assignment
"""
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
from typing import NamedTuple


@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "dvc", "pyyaml"]
)
def data_extraction(
    dataset_path: str,
    output_data: Output[Dataset]
):
    """
    Extract versioned dataset using DVC.
    
    Args:
        dataset_path: Path to the DVC-tracked dataset
        output_data: Output dataset artifact
    """
    import pandas as pd
    import os
    
    # For this assignment, we'll load the Boston Housing dataset
    from sklearn.datasets import load_boston
    
    print("Loading Boston Housing dataset...")
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['PRICE'] = boston.target
    
    # Save to output path
    df.to_csv(output_data.path, index=False)
    print(f"Dataset extracted successfully with shape: {df.shape}")
    print(f"Saved to: {output_data.path}")


@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "numpy"]
)
def data_preprocessing(
    input_data: Input[Dataset],
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Preprocess data: clean, scale, and split into train/test sets.
    
    Args:
        input_data: Input dataset
        train_data: Output training dataset
        test_data: Output testing dataset
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pickle
    
    print("Loading data for preprocessing...")
    df = pd.read_csv(input_data.path)
    
    # Separate features and target
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create DataFrames
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df['PRICE'] = y_train.values
    
    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['PRICE'] = y_test.values
    
    # Save processed data
    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)
    
    print(f"Training set shape: {train_df.shape}")
    print(f"Testing set shape: {test_df.shape}")


@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def model_training(
    train_data: Input[Dataset],
    model_output: Output[Model],
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42
):
    """
    Train a Random Forest model on the training data.
    
    Args:
        train_data: Training dataset
        model_output: Output trained model artifact
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees
        random_state: Random seed
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    
    print("Loading training data...")
    train_df = pd.read_csv(train_data.path)
    
    X_train = train_df.drop('PRICE', axis=1)
    y_train = train_df['PRICE']
    
    print(f"Training Random Forest with {n_estimators} estimators...")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, model_output.path)
    print(f"Model trained and saved to: {model_output.path}")


@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def model_evaluation(
    model_input: Input[Model],
    test_data: Input[Dataset],
    metrics_output: Output[Metrics]
) -> NamedTuple('Outputs', [('accuracy', float), ('rmse', float), ('r2_score', float)]):
    """
    Evaluate the trained model on test data.
    
    Args:
        model_input: Trained model
        test_data: Test dataset
        metrics_output: Output metrics artifact
        
    Returns:
        Tuple containing accuracy, RMSE, and R2 score
    """
    import pandas as pd
    import joblib
    from sklearn.metrics import mean_squared_error, r2_score
    import json
    import math
    
    print("Loading model and test data...")
    model = joblib.load(model_input.path)
    test_df = pd.read_csv(test_data.path)
    
    X_test = test_df.drop('PRICE', axis=1)
    y_test = test_df['PRICE']
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate accuracy (for regression, we'll use R2 as a proxy)
    accuracy = max(0, r2) * 100  # Convert to percentage
    
    metrics = {
        'accuracy': accuracy,
        'rmse': rmse,
        'r2_score': r2,
        'mse': mse
    }
    
    # Save metrics
    with open(metrics_output.path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model Evaluation Results:")
    print(f"  Accuracy (R2 %): {accuracy:.2f}%")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2 Score: {r2:.4f}")
    
    from collections import namedtuple
    output = namedtuple('Outputs', ['accuracy', 'rmse', 'r2_score'])
    return output(accuracy, rmse, r2)
