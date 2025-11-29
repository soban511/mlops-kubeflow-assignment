#!/bin/bash

# MLOps Kubeflow Assignment Setup Script

echo "========================================="
echo "MLOps Kubeflow Assignment Setup"
echo "========================================="

# Check Python version
echo "Checking Python version..."
python --version

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Initialize DVC
echo "Initializing DVC..."
dvc init

# Create data directory structure
echo "Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p components

echo "========================================="
echo "Setup completed successfully!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Configure DVC remote: dvc remote add -d myremote <path>"
echo "3. Add dataset: dvc add data/raw_data.csv"
echo "4. Compile pipeline: python pipeline.py"
echo ""
