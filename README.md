# MLOps Kubeflow Assignment

A complete Machine Learning Operations (MLOps) pipeline for Boston Housing price prediction using Kubeflow Pipelines, DVC, and Jenkins.

## Project Overview

This project demonstrates an end-to-end MLOps pipeline that includes:
- **Data Versioning** with DVC (Data Version Control)
- **Pipeline Orchestration** with Kubeflow Pipelines
- **Model Training** using Random Forest Regressor
- **Continuous Integration** with Jenkins/GitHub Actions
- **Containerization** with Docker

The ML problem: Predict housing prices in Boston using the Boston Housing dataset with 13 features including crime rate, property tax, number of rooms, etc.

## Project Structure

```
mlops-kubeflow-assignment/
├── data/                          # Data directory (DVC tracked)
│   └── raw_data.csv              # Raw dataset (tracked by DVC)
├── src/                          # Source code
│   ├── pipeline_components.py    # Kubeflow component definitions
│   └── model_training.py         # Standalone training script
├── components/                   # Compiled Kubeflow components (YAML)
├── models/                       # Saved model artifacts
├── pipeline.py                   # Main Kubeflow pipeline definition
├── pipeline.yaml                 # Compiled pipeline
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker image definition
├── Jenkinsfile                   # Jenkins CI/CD pipeline
├── .gitignore                    # Git ignore rules
├── .dvc/                         # DVC configuration
└── README.md                     # This file
```

## Prerequisites

- Python 3.9+
- Docker
- Minikube
- kubectl
- Git
- DVC

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/mlops-kubeflow-assignment.git
cd mlops-kubeflow-assignment
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. DVC Setup

Initialize DVC and configure remote storage:

```bash
# Initialize DVC
dvc init

# Configure remote storage (example with local directory)
dvc remote add -d myremote /path/to/dvc/storage

# Or use S3
# dvc remote add -d myremote s3://mybucket/dvcstore

# Pull data from remote
dvc pull
```

### 4. Minikube Setup

Start Minikube cluster:

```bash
# Start Minikube
minikube start --cpus 4 --memory 8192 --disk-size=40g

# Verify cluster is running
minikube status
kubectl cluster-info
```

### 5. Kubeflow Pipelines Installation

Deploy Kubeflow Pipelines on Minikube:

```bash
# Install Kubeflow Pipelines standalone
export PIPELINE_VERSION=2.0.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$PIPELINE_VERSION"

# Wait for pods to be ready
kubectl wait --for=condition=ready --timeout=300s pods --all -n kubeflow

# Access the UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

Access the Kubeflow Pipelines UI at: http://localhost:8080

## Pipeline Walkthrough

### Step 1: Compile the Pipeline

```bash
python pipeline.py
```

This generates `pipeline.yaml` containing the compiled pipeline definition.

### Step 2: Upload and Run Pipeline

1. Open Kubeflow Pipelines UI (http://localhost:8080)
2. Click "Upload Pipeline"
3. Select `pipeline.yaml`
4. Create a new run with default parameters
5. Monitor the execution

### Pipeline Components

The pipeline consists of 4 main components:

1. **Data Extraction**: Loads the Boston Housing dataset
   - Input: Dataset path
   - Output: Raw dataset artifact

2. **Data Preprocessing**: Cleans, scales, and splits data
   - Input: Raw dataset
   - Output: Training and testing datasets

3. **Model Training**: Trains Random Forest model
   - Input: Training dataset, hyperparameters
   - Output: Trained model artifact

4. **Model Evaluation**: Evaluates model performance
   - Input: Trained model, test dataset
   - Output: Metrics (RMSE, R2 score, accuracy)

### Local Testing

Test the training script locally:

```bash
python src/model_training.py
```

## Continuous Integration

### Jenkins Setup

1. Install Jenkins and required plugins (Git, Pipeline)
2. Create a new Pipeline job
3. Configure SCM to point to this repository
4. Set script path to `Jenkinsfile`
5. Trigger build manually or via webhook

The Jenkins pipeline includes:
- **Environment Setup**: Install dependencies
- **Pipeline Compilation**: Compile Kubeflow pipeline
- **Validation**: Validate pipeline YAML

### GitHub Actions (Alternative)

Create `.github/workflows/pipeline.yml` for automated CI/CD on push.

## Data Versioning with DVC

### Track New Data

```bash
# Add data file
dvc add data/raw_data.csv

# Commit DVC file
git add data/raw_data.csv.dvc data/.gitignore
git commit -m "Track dataset with DVC"

# Push data to remote
dvc push
```

### Pull Data

```bash
dvc pull
```

### Check Status

```bash
dvc status
```

## Model Performance

Expected metrics on Boston Housing dataset:
- **R2 Score**: ~0.85-0.90
- **RMSE**: ~3.0-4.0
- **Accuracy**: 85-90%

## Troubleshooting

### Minikube Issues

```bash
# Restart Minikube
minikube stop
minikube start

# Check logs
minikube logs
```

### Kubeflow Pipeline Issues

```bash
# Check pod status
kubectl get pods -n kubeflow

# View logs
kubectl logs -n kubeflow <pod-name>
```

### DVC Issues

```bash
# Check DVC status
dvc status

# Verify remote configuration
dvc remote list
```

## Technologies Used

- **Python 3.9**: Programming language
- **Kubeflow Pipelines**: ML workflow orchestration
- **DVC**: Data version control
- **Scikit-learn**: Machine learning library
- **Docker**: Containerization
- **Kubernetes/Minikube**: Container orchestration
- **Jenkins**: CI/CD automation
- **Git/GitHub**: Version control

## Contributors

- Your Name - MLOps Assignment

## License

This project is for educational purposes as part of the MLOps course assignment.

## References

- [Kubeflow Pipelines Documentation](https://www.kubeflow.org/docs/components/pipelines/)
- [DVC Documentation](https://dvc.org/doc)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Minikube Documentation](https://minikube.sigs.k8s.io/docs/)
