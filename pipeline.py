"""
Kubeflow Pipeline Definition for MLOps Assignment
"""
from kfp import dsl
from kfp import compiler
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pipeline_components import (
    data_extraction,
    data_preprocessing,
    model_training,
    model_evaluation
)


@dsl.pipeline(
    name='Boston Housing ML Pipeline',
    description='End-to-end ML pipeline for Boston Housing price prediction'
)
def boston_housing_pipeline(
    dataset_path: str = 'data/raw_data.csv',
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42
):
    """
    Complete ML pipeline with data extraction, preprocessing, training, and evaluation.
    
    Args:
        dataset_path: Path to the dataset
        test_size: Proportion of data for testing
        n_estimators: Number of trees in Random Forest
        max_depth: Maximum depth of trees
        random_state: Random seed for reproducibility
    """
    
    # Step 1: Data Extraction
    extract_task = data_extraction(
        dataset_path=dataset_path
    )
    extract_task.set_display_name('Extract Data')
    
    # Step 2: Data Preprocessing
    preprocess_task = data_preprocessing(
        input_data=extract_task.outputs['output_data'],
        test_size=test_size,
        random_state=random_state
    )
    preprocess_task.set_display_name('Preprocess Data')
    preprocess_task.after(extract_task)
    
    # Step 3: Model Training
    train_task = model_training(
        train_data=preprocess_task.outputs['train_data'],
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    train_task.set_display_name('Train Model')
    train_task.after(preprocess_task)
    
    # Step 4: Model Evaluation
    eval_task = model_evaluation(
        model_input=train_task.outputs['model_output'],
        test_data=preprocess_task.outputs['test_data']
    )
    eval_task.set_display_name('Evaluate Model')
    eval_task.after(train_task)


def compile_pipeline():
    """Compile the pipeline to YAML"""
    output_file = 'pipeline.yaml'
    
    print("Compiling Kubeflow pipeline...")
    compiler.Compiler().compile(
        pipeline_func=boston_housing_pipeline,
        package_path=output_file
    )
    print(f"Pipeline compiled successfully to {output_file}")


if __name__ == '__main__':
    compile_pipeline()
