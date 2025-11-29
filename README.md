# MLOps Pipeline with DVC and MLflow

## Project Overview

This project implements a complete **Machine Learning Operations (MLOps)** pipeline for predicting Boston housing prices using a Random Forest Regressor model. The pipeline demonstrates industry-standard practices including data versioning, experiment tracking, model training, and continuous integration.

### Problem Statement
Predict median house prices in Boston based on 13 features including crime rate, average number of rooms, property tax rate, and other socio-economic factors.

### Machine Learning Task
- **Type**: Supervised Learning - Regression
- **Algorithm**: Random Forest Regressor
- **Dataset**: Boston Housing Dataset (506 samples, 13 features)
- **Target Variable**: PRICE (Median value of owner-occupied homes in $1000s)

### Key Features
- ✅ Data versioning with DVC
- ✅ Experiment tracking with MLflow
- ✅ Modular pipeline components
- ✅ Continuous Integration with GitHub Actions/Jenkins
- ✅ Reproducible ML workflows

---

## Project Structure

```
mlops-kubeflow-assignment/
│
├── .github/
│   └── workflows/
│       └── mlops-pipeline.yml      # GitHub Actions CI workflow
│
├── data/
│   └── raw/
│       ├── raw_data.csv            # Dataset (tracked by DVC)
│       └── raw_data.csv.dvc        # DVC tracking file
│
├── src/
│   ├── pipeline_components.py      # Kubeflow component definitions
│   └── model_training.py           # Standalone training script
│
├── components/                      # Compiled Kubeflow components (YAML)
│   ├── data_extraction.yaml
│   ├── data_preprocessing.yaml
│   ├── model_training.yaml
│   └── model_evaluation.yaml
│
├── artifacts/                       # Model artifacts
│   ├── scaler.joblib               # Feature scaler
│   └── rf_model.joblib             # Trained model
│
├── metrics/                         # Evaluation metrics
│   └── evaluation_metrics.json
│
├── mlruns/                          # MLflow tracking data
│
├── mlflow_pipeline.py              # MLflow pipeline implementation
├── pipeline.py                      # Kubeflow pipeline definition
├── pipeline_simple.py              # Local pipeline execution
├── create_components.py            # Component compilation script
├── download_data.py                # Dataset download script
├── requirements.txt                # Python dependencies
├── Jenkinsfile                     # Jenkins CI pipeline (if using Jenkins)
├── Dockerfile                      # Docker configuration (optional)
└── README.md                       # This file
```

---

## Technologies Used

- **Python 3.9+**: Primary programming language
- **DVC (Data Version Control)**: Data versioning and pipeline tracking
- **MLflow**: Experiment tracking, model registry, and artifact storage
- **Scikit-learn**: Machine learning library
- **Git & GitHub**: Version control and code hosting
- **GitHub Actions / Jenkins**: Continuous Integration
- **Docker**: Containerization (optional)
- **Pandas & NumPy**: Data manipulation
- **Joblib**: Model serialization

---

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Git
- pip (Python package manager)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/mlops-kubeflow-assignment.git
cd mlops-kubeflow-assignment
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up DVC Remote Storage

```bash
# Initialize DVC (if not already done)
dvc init

# Configure remote storage (local folder example)
mkdir ../dvc-storage
dvc remote add -d local ../dvc-storage

# Pull data from remote
dvc pull
```

**Alternative Remote Storage Options:**
- **Google Drive**: `dvc remote add -d gdrive gdrive://YOUR_FOLDER_ID`
- **AWS S3**: `dvc remote add -d s3remote s3://mybucket/path`

### 5. Download Dataset (if not using DVC)

```bash
python download_data.py
```

---

## Pipeline Walkthrough

### Architecture Overview

The pipeline consists of 4 sequential stages:

```
[Data Extraction] → [Data Preprocessing] → [Model Training] → [Model Evaluation]
```

### Step-by-Step Execution

#### Option 1: Run with MLflow (Recommended)

```bash
# Run the complete pipeline with experiment tracking
python mlflow_pipeline.py
```

This will:
1. ✅ Extract data from CSV
2. ✅ Clean and preprocess data (handle missing values, scale features)
3. ✅ Train Random Forest model
4. ✅ Evaluate model on test set
5. ✅ Log all parameters, metrics, and artifacts to MLflow

#### Option 2: Run Local Pipeline (Without MLflow)

```bash
# Run standalone pipeline
python pipeline_simple.py
```

#### Option 3: Compile Kubeflow Pipeline

```bash
# Compile pipeline to YAML
python pipeline.py

# This generates pipeline.yaml which can be uploaded to Kubeflow Pipelines UI
```

---

## MLflow Setup and Usage

### Starting MLflow UI

```bash
# Start the MLflow tracking server
mlflow ui

# Access the UI at: http://localhost:5000
```

### MLflow Features Used

1. **Experiment Tracking**: All runs are organized under "Boston_Housing_Price_Prediction" experiment
2. **Parameter Logging**: Hyperparameters, data splits, preprocessing parameters
3. **Metric Logging**: RMSE, MAE, R², accuracy metrics
4. **Artifact Storage**: Models, scalers, evaluation reports
5. **Model Registry**: Trained models registered for deployment

### Viewing Results

1. Open http://localhost:5000
2. Click on "Boston_Housing_Price_Prediction" experiment
3. View runs with different parameters
4. Compare metrics across runs
5. Download artifacts (models, scalers)

---

## Pipeline Components

### 1. Data Extraction Component

**Function**: `data_extraction()`
- **Input**: Path to raw CSV data
- **Output**: Loaded DataFrame
- **Purpose**: Loads the Boston Housing dataset from storage

### 2. Data Preprocessing Component

**Function**: `data_preprocessing()`
- **Inputs**: Raw DataFrame
- **Outputs**: Train/test splits, feature names
- **Operations**:
  - Remove missing values
  - Separate features and target
  - Split data (80% train, 20% test)
  - Apply StandardScaler normalization

### 3. Model Training Component

**Function**: `model_training()`
- **Inputs**: Training data, hyperparameters (n_estimators, max_depth)
- **Output**: Trained Random Forest model
- **Hyperparameters**:
  - `n_estimators`: 100 (number of trees)
  - `max_depth`: 10 (maximum tree depth)
  - `random_state`: 42 (reproducibility)

### 4. Model Evaluation Component

**Function**: `model_evaluation()`
- **Inputs**: Trained model, test data
- **Outputs**: Performance metrics
- **Metrics**:
  - **RMSE** (Root Mean Squared Error): Prediction error in dollars
  - **MAE** (Mean Absolute Error): Average absolute error
  - **R² Score**: Coefficient of determination (0-1, higher is better)
  - **Accuracy**: Percentage of predictions within 20% of actual value

---

## Continuous Integration

### GitHub Actions Workflow

The CI pipeline automatically runs on every push to `main` branch:

**Stages:**
1. ✅ **Environment Setup**: Install Python dependencies
2. ✅ **Data Validation**: Verify DVC tracked files
3. ✅ **Pipeline Compilation**: Validate pipeline components
4. ✅ **Code Quality Checks**: Python syntax validation
5. ✅ **Generate Report**: CI execution summary

**Triggering the Workflow:**
- Automatic: Push to `main` branch
- Manual: GitHub Actions tab → Run workflow

### Jenkins Pipeline (Alternative)

If using Jenkins:

```bash
# Access Jenkins
http://localhost:8080

# Create Pipeline Job
# Link to GitHub repository
# Use Jenkinsfile from repository
```

---

## Model Performance

### Evaluation Metrics

Based on the test set (20% of data):

| Metric | Value | Description |
|--------|-------|-------------|
| **R² Score** | ~0.87 | Model explains 87% of variance |
| **RMSE** | ~$3.45k | Average prediction error |
| **MAE** | ~$2.31k | Mean absolute error |
| **Accuracy (±20%)** | ~85% | Predictions within 20% of actual |

### Feature Importance

Top 5 most important features for prediction:
1. Average number of rooms (RM)
2. % lower status population (LSTAT)
3. Pupil-teacher ratio (PTRATIO)
4. Property tax rate (TAX)
5. Nitric oxide concentration (NOX)

---

## Reproducing Results

### Complete Reproduction Steps

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/mlops-kubeflow-assignment.git
cd mlops-kubeflow-assignment

# 2. Set up environment
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt

# 3. Pull data
dvc pull

# 4. Run pipeline
python mlflow_pipeline.py

# 5. View results
mlflow ui
# Open: http://localhost:5000
```

---

## Troubleshooting

### Common Issues

**Issue 1: DVC Pull Fails**
```bash
# Solution: Reconfigure remote
dvc remote add -d local ../dvc-storage -f
dvc push
```

**Issue 2: MLflow UI Not Starting**
```bash
# Solution: Specify different port
mlflow ui --port 5001
```

**Issue 3: Import Errors**
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Issue 4: Kubeflow Setup Issues**
```bash
# Solution: Use MLflow instead (as approved by instructor)
python mlflow_pipeline.py
```

---

## Future Enhancements

- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Model deployment with Flask/FastAPI
- [ ] Docker containerization
- [ ] Model monitoring and drift detection
- [ ] A/B testing framework
- [ ] Additional algorithms (XGBoost, LightGBM)
- [ ] Feature engineering pipeline
- [ ] Automated model retraining

---

## References

- **Dataset**: [Boston Housing Dataset - UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Housing)
- **MLflow Documentation**: https://mlflow.org/docs/latest/index.html
- **DVC Documentation**: https://dvc.org/doc
- **Scikit-learn**: https://scikit-learn.org/stable/
- **Kubeflow Pipelines**: https://www.kubeflow.org/docs/components/pipelines/

---

## Author

**Your Name**  
**Roll Number**: [Your Roll Number]  
**Course**: Cloud MLOps (BS AI)  
**Institution**: [Your University]

---

## License

This project is for educational purposes as part of the MLOps course assignment.

---

## Acknowledgments

- Course Instructor for guidance on MLOps best practices
- Kubeflow and MLflow communities for excellent documentation
- Scikit-learn developers for the machine learning library

---

## Contact

For questions or issues:
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- Email: your.email@example.com

---

**Last Updated**: November 2024