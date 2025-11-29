# Quick Start Guide

Follow these steps to get your MLOps pipeline up and running quickly.

## Step 1: Initial Setup (5 minutes)

### Windows:
```cmd
setup.bat
```

### Linux/Mac:
```bash
chmod +x setup.sh
./setup.sh
```

## Step 2: Generate Dataset (1 minute)

```bash
# Activate virtual environment first
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

python src/generate_dataset.py
```

## Step 3: Initialize DVC (2 minutes)

```bash
# Initialize DVC (if not done by setup script)
dvc init

# Create a local remote storage
mkdir ../dvc-storage

# Add remote
dvc remote add -d myremote ../dvc-storage

# Track the dataset
dvc add data/raw_data.csv

# Commit DVC files
git add data/raw_data.csv.dvc data/.gitignore .dvc/config
git commit -m "Add dataset with DVC tracking"

# Push data to remote
dvc push
```

## Step 4: Test Locally (2 minutes)

```bash
# Test the training script
python src/model_training.py

# This should create a models/ directory with:
# - random_forest_model.pkl
# - scaler.pkl
# - metrics.json
```

## Step 5: Compile Kubeflow Pipeline (1 minute)

```bash
python pipeline.py
```

This creates `pipeline.yaml` file.

## Step 6: Setup Minikube (10 minutes)

```bash
# Start Minikube
minikube start --cpus 4 --memory 8192

# Verify
minikube status
kubectl get nodes
```

## Step 7: Install Kubeflow Pipelines (15 minutes)

```bash
# Set version
export PIPELINE_VERSION=2.0.0

# Install cluster-scoped resources
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"

# Wait for CRDs
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

# Install Kubeflow Pipelines
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$PIPELINE_VERSION"

# Wait for all pods to be ready
kubectl wait --for=condition=ready --timeout=300s pods --all -n kubeflow

# Port forward to access UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

## Step 8: Run Pipeline (5 minutes)

1. Open browser: http://localhost:8080
2. Click "Upload Pipeline"
3. Select `pipeline.yaml`
4. Click "Create Run"
5. Use default parameters
6. Click "Start"
7. Watch the pipeline execute!

## Step 9: Setup Jenkins (Optional, 15 minutes)

### Install Jenkins:

**Windows:**
- Download from: https://www.jenkins.io/download/
- Install and start Jenkins
- Access: http://localhost:8080

**Linux:**
```bash
# Install Java
sudo apt update
sudo apt install openjdk-11-jdk

# Install Jenkins
wget -q -O - https://pkg.jenkins.io/debian/jenkins.io.key | sudo apt-key add -
sudo sh -c 'echo deb http://pkg.jenkins.io/debian-stable binary/ > /etc/apt/sources.list.d/jenkins.list'
sudo apt update
sudo apt install jenkins

# Start Jenkins
sudo systemctl start jenkins
```

### Configure Jenkins:
1. Access Jenkins UI
2. Install suggested plugins
3. Create admin user
4. Create new Pipeline job
5. Configure Git repository
6. Set script path to `Jenkinsfile`
7. Build the project

## Step 10: Push to GitHub

```bash
# Add all files
git add .

# Commit
git commit -m "Complete MLOps pipeline implementation"

# Push to GitHub
git push origin main
```

## Verification Checklist

- [ ] Virtual environment created and activated
- [ ] Dependencies installed
- [ ] Dataset generated in `data/raw_data.csv`
- [ ] DVC initialized and remote configured
- [ ] Dataset tracked with DVC
- [ ] Local training script works
- [ ] Pipeline compiled to `pipeline.yaml`
- [ ] Minikube running
- [ ] Kubeflow Pipelines installed
- [ ] Pipeline uploaded and executed successfully
- [ ] Jenkins configured (optional)
- [ ] All code pushed to GitHub

## Troubleshooting

### Issue: Python package conflicts
**Solution:** Use a fresh virtual environment

### Issue: Minikube won't start
**Solution:** 
```bash
minikube delete
minikube start --driver=docker
```

### Issue: Kubeflow pods not ready
**Solution:**
```bash
kubectl get pods -n kubeflow
kubectl describe pod <pod-name> -n kubeflow
```

### Issue: DVC push fails
**Solution:** Check remote configuration
```bash
dvc remote list
dvc remote modify myremote url /new/path
```

## Next Steps

1. Experiment with different hyperparameters
2. Try different ML models
3. Add more pipeline components
4. Implement model monitoring
5. Set up automated retraining

## Need Help?

- Check the main README.md for detailed documentation
- Review Kubeflow Pipelines docs: https://www.kubeflow.org/docs/
- DVC documentation: https://dvc.org/doc
