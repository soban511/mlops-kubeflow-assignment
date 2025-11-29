"""
MLflow Pipeline for Boston Housing Price Prediction
This pipeline tracks all experiments, parameters, and metrics using MLflow.
Each step is tracked as a nested run for full visibility.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import json


def data_extraction(data_path):
    """
    Step 1: Data Extraction
    Load the dataset from CSV file and log metadata to MLflow
    """
    print("\n" + "="*70)
    print("[STEP 1/4] DATA EXTRACTION")
    print("="*70)
    
    with mlflow.start_run(run_name="01_data_extraction", nested=True):
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        # Log parameters
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("num_samples", len(df))
        mlflow.log_param("num_features", len(df.columns) - 1)
        mlflow.log_param("target_variable", "PRICE")
        
        # Log metrics
        mlflow.log_metric("dataset_size_mb", df.memory_usage(deep=True).sum() / 1024 / 1024)
        
        print(f"✓ Data loaded successfully")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        return df


def data_preprocessing(df, test_size=0.2, random_state=42):
    """
    Step 2: Data Preprocessing
    Clean, scale, and split the data into train/test sets
    """
    print("\n" + "="*70)
    print("[STEP 2/4] DATA PREPROCESSING")
    print("="*70)
    
    with mlflow.start_run(run_name="02_data_preprocessing", nested=True):
        # Handle missing values
        initial_rows = len(df)
        df_clean = df.dropna()
        dropped_rows = initial_rows - len(df_clean)
        
        # Log preprocessing parameters
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("scaling_method", "StandardScaler")
        mlflow.log_param("dropped_rows", dropped_rows)
        
        print(f"✓ Data cleaning completed")
        print(f"  Rows dropped (NaN): {dropped_rows}")
        print(f"  Clean data shape: {df_clean.shape}")
        
        # Separate features and target
        X = df_clean.drop('PRICE', axis=1)
        y = df_clean['PRICE']
        
        feature_names = X.columns.tolist()
        mlflow.log_param("num_features", len(feature_names))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Log split statistics
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_metric("train_target_mean", y_train.mean())
        mlflow.log_metric("test_target_mean", y_test.mean())
        
        print(f"✓ Data split completed")
        print(f"  Train set: {len(X_train)} samples ({len(X_train)/len(df_clean)*100:.1f}%)")
        print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(df_clean)*100:.1f}%)")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"✓ Feature scaling completed (StandardScaler)")
        
        # Save scaler artifact
        os.makedirs('artifacts', exist_ok=True)
        scaler_path = 'artifacts/scaler.joblib'
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="preprocessor")
        
        print(f"✓ Scaler saved to: {scaler_path}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_names


def model_training(X_train, y_train, feature_names, n_estimators=100, max_depth=10):
    """
    Step 3: Model Training
    Train a Random Forest Regressor model
    """
    print("\n" + "="*70)
    print("[STEP 3/4] MODEL TRAINING")
    print("="*70)
    
    with mlflow.start_run(run_name="03_model_training", nested=True):
        # Log hyperparameters
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_jobs", -1)
        
        print(f"Training Random Forest Regressor...")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_depth: {max_depth}")
        print(f"  random_state: 42")
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        print(f"✓ Model training completed")
        
        # Get feature importances
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n✓ Top 10 Important Features:")
        for i, (feat, importance) in enumerate(top_features[:10], 1):
            print(f"  {i:2d}. {feat:15s}: {importance:.4f}")
            mlflow.log_metric(f"feature_importance_{feat}", importance)
        
        # Log top 5 feature names as params
        for i, (feat, _) in enumerate(top_features[:5], 1):
            mlflow.log_param(f"top_feature_{i}", feat)
        
        # Save and log model
        model_path = 'artifacts/rf_model.joblib'
        joblib.dump(model, model_path)
        
        # Log model with MLflow (enables model serving)
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="BostonHousingRFModel"
        )
        mlflow.log_artifact(model_path, artifact_path="model_files")
        
        print(f"✓ Model saved to: {model_path}")
        print(f"✓ Model registered in MLflow Model Registry")
        
        return model


def model_evaluation(model, X_test, y_test):
    """
    Step 4: Model Evaluation
    Evaluate the trained model on test data
    """
    print("\n" + "="*70)
    print("[STEP 4/4] MODEL EVALUATION")
    print("="*70)
    
    with mlflow.start_run(run_name="04_model_evaluation", nested=True):
        print("Making predictions on test set...")
        y_pred = model.predict(X_test)
        
        # Calculate comprehensive metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        accuracy_20 = np.mean(np.abs((y_test - y_pred) / y_test) < 0.2) * 100
        accuracy_10 = np.mean(np.abs((y_test - y_pred) / y_test) < 0.1) * 100
        
        # Log all metrics to MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("accuracy_within_20_percent", accuracy_20)
        mlflow.log_metric("accuracy_within_10_percent", accuracy_10)
        mlflow.log_metric("mean_prediction", y_pred.mean())
        mlflow.log_metric("std_prediction", y_pred.std())
        
        # Create metrics dictionary
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'mse': float(mse),
            'r2_score': float(r2),
            'mape': float(mape),
            'accuracy_within_20_percent': float(accuracy_20),
            'accuracy_within_10_percent': float(accuracy_10),
            'num_test_samples': len(y_test),
            'mean_actual_price': float(y_test.mean()),
            'std_actual_price': float(y_test.std()),
            'mean_predicted_price': float(y_pred.mean()),
            'std_predicted_price': float(y_pred.std())
        }
        
        # Save metrics to JSON
        os.makedirs('metrics', exist_ok=True)
        metrics_path = 'metrics/evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        mlflow.log_artifact(metrics_path, artifact_path="evaluation")
        
        # Display results
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"Root Mean Squared Error (RMSE):  ${rmse:,.2f}")
        print(f"Mean Absolute Error (MAE):       ${mae:,.2f}")
        print(f"Mean Squared Error (MSE):        {mse:,.2f}")
        print(f"R² Score (Coefficient of Det.):  {r2:.4f}")
        print(f"Mean Absolute % Error (MAPE):    {mape:.2f}%")
        print(f"Accuracy (within 20% of actual): {accuracy_20:.2f}%")
        print(f"Accuracy (within 10% of actual): {accuracy_10:.2f}%")
        print(f"Number of test samples:          {len(y_test)}")
        print("="*70)
        
        print(f"\n✓ Metrics saved to: {metrics_path}")
        
        return metrics


def run_mlflow_pipeline(data_path='data/raw/raw_data.csv', 
                       n_estimators=100, 
                       max_depth=10,
                       experiment_name="Boston_Housing_Price_Prediction"):
    """
    Run the complete MLflow pipeline with all 4 steps
    
    Args:
        data_path: Path to the raw data CSV file
        n_estimators: Number of trees in Random Forest
        max_depth: Maximum depth of trees
        experiment_name: Name of the MLflow experiment
    
    Returns:
        metrics: Dictionary containing all evaluation metrics
    """
    # Set experiment (creates if doesn't exist)
    mlflow.set_experiment(experiment_name)
    
    print("\n" + "="*70)
    print("BOSTON HOUSING PRICE PREDICTION - MLFLOW PIPELINE")
    print("="*70)
    print(f"Experiment Name: {experiment_name}")
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Data Path: {data_path}")
    print(f"Hyperparameters: n_estimators={n_estimators}, max_depth={max_depth}")
    print("="*70)
    
    # Start parent run (contains all nested runs)
    with mlflow.start_run(run_name="complete_ml_pipeline"):
        # Log high-level pipeline parameters
        mlflow.log_param("pipeline_version", "1.0.0")
        mlflow.log_param("data_source", data_path)
        mlflow.log_param("model_hyperparams", f"n_estimators={n_estimators}, max_depth={max_depth}")
        mlflow.set_tag("pipeline_type", "supervised_regression")
        mlflow.set_tag("framework", "scikit-learn")
        mlflow.set_tag("dataset", "Boston Housing")
        
        # Step 1: Data Extraction
        df = data_extraction(data_path)
        
        # Step 2: Data Preprocessing
        X_train, X_test, y_train, y_test, feature_names = data_preprocessing(df)
        
        # Step 3: Model Training
        model = model_training(X_train, y_train, feature_names, n_estimators, max_depth)
        
        # Step 4: Model Evaluation
        metrics = model_evaluation(model, X_test, y_test)
        
        # Log final pipeline status
        mlflow.log_param("pipeline_status", "SUCCESS")
        
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nAll runs logged to MLflow")
        print("\nTo view results in MLflow UI:")
        print("  1. Run: mlflow ui")
        print("  2. Open: http://localhost:5000")
        print("\nArtifacts saved:")
        print("  - artifacts/scaler.joblib")
        print("  - artifacts/rf_model.joblib")
        print("  - metrics/evaluation_metrics.json")
        print("="*70 + "\n")
        
        return metrics


if __name__ == "__main__":
    # Run the complete pipeline
    print("Starting MLflow Pipeline Execution...")
    
    metrics = run_mlflow_pipeline(
        data_path='data/raw/raw_data.csv',
        n_estimators=100,
        max_depth=10,
        experiment_name="Boston_Housing_Price_Prediction"
    )
    
    print("\nPipeline execution completed!")
    print(f"Final R² Score: {metrics['r2_score']:.4f}")
    print(f"Final RMSE: ${metrics['rmse']:.2f}")