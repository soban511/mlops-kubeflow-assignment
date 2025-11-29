"""
Script to compile Kubeflow pipeline components to YAML files.
This allows components to be versioned and reused across different pipelines.
"""

from kfp import compiler
from src.pipeline_components import (
    data_extraction,
    data_preprocessing,
    model_training,
    model_evaluation
)
import os


def main():
    """Compile all pipeline components to YAML files."""
    
    # Create components directory if it doesn't exist
    os.makedirs('components', exist_ok=True)
    
    print("="*60)
    print("COMPILING KUBEFLOW PIPELINE COMPONENTS")
    print("="*60)
    
    # Compile each component to YAML
    components = [
        (data_extraction, 'data_extraction.yaml', 'Data Extraction'),
        (data_preprocessing, 'data_preprocessing.yaml', 'Data Preprocessing'),
        (model_training, 'model_training.yaml', 'Model Training'),
        (model_evaluation, 'model_evaluation.yaml', 'Model Evaluation')
    ]
    
    for component_func, yaml_file, component_name in components:
        output_path = f'components/{yaml_file}'
        print(f"\nCompiling: {component_name}")
        print(f"Output: {output_path}")
        
        try:
            compiler.Compiler().compile(
                component_func,
                output_path
            )
            print(f"✓ Successfully compiled {yaml_file}")
        except Exception as e:
            print(f"✗ Error compiling {yaml_file}: {str(e)}")
            raise
    
    print("\n" + "="*60)
    print("ALL COMPONENTS COMPILED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated YAML files in 'components/' directory:")
    for _, yaml_file, _ in components:
        print(f"  - {yaml_file}")
    print("\nYou can now use these components in your pipeline definition.")


if __name__ == "__main__":
    main()