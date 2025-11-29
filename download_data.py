import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml

# Fetch Boston housing dataset
print("Downloading Boston Housing dataset...")
boston = fetch_openml(name='boston', version=1, parser='auto')

# Create DataFrame
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

# Save to CSV
df.to_csv('data/raw/raw_data.csv', index=False)
print(f"Dataset saved with shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")