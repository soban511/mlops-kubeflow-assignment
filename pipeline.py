"""
Kubeflow Pipeline Definition for Boston Housing Price Prediction
This file defines the complete ML pipeline by connecting all components.
"""

from kfp import dsl, compiler
from src.pipeline_components import (
    data_extraction,
    data_preprocessing,
    model_training,
    model_evaluation
)


@dsl.pipeline(
    name='Boston Housing Price Prediction Pipeline',
    description='End-to-end ML pipeline for predicting Boston housing prices using Random Forest'
)
def boston_housing_pipeline(
    data_path: str = '/data/raw_data.csv',
    n_estimators: int = 100,
    max_depth: int = 10
):
    """
    Complete ML Pipeline for Boston Housing Price Prediction.
    
    Pipeline Steps:
    1. Data Extraction: Load the dataset
    2. Data Preprocessing: Clean, scale, and split data
    3. Model Training: Train Random Forest model
    4. Model Evaluation: Evaluate model performance
    
    Args:
        data_path: Path to the raw dataset CSV file
        n_estimators: Number of trees in Random Forest (default: 100)
        max_depth: Maximum depth of trees (default: 10)
    """
    
    # Step 1: Data Extraction
    extraction_task = data_extraction(
        data_path=data_path
    )
    extraction_task.set_display_name('Extract Data')
    
    # Step 2: Data Preprocessing
    preprocessing_task = data_preprocessing(
        input_data_path=extraction_task.output
    )
    preprocessing_task.set_display_name('Preprocess Data')
    preprocessing_task.after(extraction_task)
    
    # Step 3: Model Training
    training_task = model_training(
        train_data_path=preprocessing_task.outputs['train_data_path'],
        n_estimators=n_estimators,
        max_depth=max_depth
    )
    training_task.set_display_name('Train Model')
    training_task.after(preprocessing_task)
    
    # Step 4: Model Evaluation
    evaluation_task = model_evaluation(
        model_path=training_task.output,
        test_data_path=preprocessing_task.outputs['test_data_path']
    )
    evaluation_task.set_display_name('Evaluate Model')
    evaluation_task.after(training_task)
    
    # Print outputs for visibility
    print(f"Pipeline created with parameters:")
    print(f"  - Data path: {data_path}")
    print(f"  - Number of estimators: {n_estimators}")
    print(f"  - Max depth: {max_depth}")


def compile_pipeline(output_path: str = 'pipeline.yaml'):
    """
    Compile the pipeline to a YAML file.
    
    Args:
        output_path: Path where the compiled pipeline YAML will be saved
    """
    print("="*60)
    print("COMPILING KUBEFLOW PIPELINE")
    print("="*60)
    print(f"\nPipeline Name: Boston Housing Price Prediction Pipeline")
    print(f"Output file: {output_path}")
    print("\nCompiling...")
    
    try:
        compiler.Compiler().compile(
            pipeline_func=boston_housing_pipeline,
            package_path=output_path
        )
        print(f"\n✓ Pipeline compiled successfully!")
        print(f"✓ YAML file saved to: {output_path}")
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Ensure Minikube is running: minikube status")
        print("2. Port-forward KFP UI: kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80")
        print("3. Open browser: http://localhost:8080")
        print("4. Upload and run the pipeline.yaml file")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error compiling pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    # Compile the pipeline when this script is run directly
    compile_pipeline('pipeline.yaml')