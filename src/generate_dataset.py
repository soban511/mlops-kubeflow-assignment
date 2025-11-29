"""
Generate Boston Housing dataset for the project
"""
import pandas as pd
import os


def generate_boston_dataset():
    """Generate and save the Boston Housing dataset"""
    try:
        from sklearn.datasets import load_boston
        print("Loading Boston Housing dataset...")
        boston = load_boston()
        
        # Create DataFrame
        df = pd.DataFrame(boston.data, columns=boston.feature_names)
        df['PRICE'] = boston.target
        
    except ImportError:
        print("Warning: load_boston is deprecated. Using alternative dataset...")
        # Alternative: Create a sample dataset or use California Housing
        from sklearn.datasets import fetch_california_housing
        california = fetch_california_housing()
        df = pd.DataFrame(california.data, columns=california.feature_names)
        df['PRICE'] = california.target
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save dataset
    output_path = 'data/raw_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to: {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nDataset info:")
    print(df.info())
    
    return df


if __name__ == "__main__":
    generate_boston_dataset()
