# Useful Commands Reference

Quick reference for all commands you'll need during the assignment.

## Git Commands

```bash
# Initial setup
git init
git remote add origin https://github.com/YOUR_USERNAME/mlops-kubeflow-assignment.git

# Daily workflow
git status
git add .
git commit -m "Your message"
git push origin main

# View history
git log --oneline
```

## Python & Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Deactivate
deactivate
```

## DVC Commands

```bash
# Initialize DVC
dvc init

# Add remote storage
dvc remote add -d myremote /path/to/storage
dvc remote add -d myremote s3://bucket/path
dvc remote add -d myremote gdrive://folder_id

# List remotes
dvc remote list

# Track data
dvc add data/raw_data.csv

# Push data to remote
dvc push

# Pull data from remote
dvc pull

# Check status
dvc status

# Remove tracking
dvc remove data/raw_data.csv.dvc
```

## Minikube Commands

```bash
# Start Minikube
minikube start
minikube start --cpus 4 --memory 8192 --disk-size=40g

# Check status
minikube status

# Stop Minikube
minikube stop

# Delete cluster
minikube delete

# Access dashboard
minikube dashboard

# SSH into node
minikube ssh

# View logs
minikube logs

# Get IP
minikube ip
```

## Kubectl Commands

```bash
# Cluster info
kubectl cluster-info
kubectl get nodes

# Namespaces
kubectl get namespaces
kubectl get ns

# Pods
kubectl get pods -n kubeflow
kubectl get pods --all-namespaces
kubectl describe pod <pod-name> -n kubeflow
kubectl logs <pod-name> -n kubeflow
kubectl logs -f <pod-name> -n kubeflow  # Follow logs

# Services
kubectl get services -n kubeflow
kubectl get svc -n kubeflow

# Port forwarding
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80

# Delete resources
kubectl delete pod <pod-name> -n kubeflow
kubectl delete namespace kubeflow

# Apply manifests
kubectl apply -f manifest.yaml
kubectl apply -k <directory>

# Wait for resources
kubectl wait --for=condition=ready --timeout=300s pods --all -n kubeflow
```

## Kubeflow Pipelines Installation

```bash
# Set version
export PIPELINE_VERSION=2.0.0

# Install cluster-scoped resources
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"

# Wait for CRDs
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

# Install Kubeflow Pipelines
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$PIPELINE_VERSION"

# Wait for pods
kubectl wait --for=condition=ready --timeout=300s pods --all -n kubeflow

# Access UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

## Pipeline Commands

```bash
# Generate dataset
python src/generate_dataset.py

# Test training locally
python src/model_training.py

# Compile pipeline
python pipeline.py

# Validate pipeline YAML
python -c "import yaml; yaml.safe_load(open('pipeline.yaml'))"
```

## Docker Commands

```bash
# Build image
docker build -t mlops-pipeline:latest .

# List images
docker images

# Run container
docker run -it mlops-pipeline:latest

# Remove image
docker rmi mlops-pipeline:latest

# Clean up
docker system prune -a
```

## Jenkins Commands

```bash
# Start Jenkins (Linux)
sudo systemctl start jenkins
sudo systemctl status jenkins

# Stop Jenkins
sudo systemctl stop jenkins

# Restart Jenkins
sudo systemctl restart jenkins

# View logs
sudo journalctl -u jenkins -f
```

## Troubleshooting Commands

```bash
# Check Python version
python --version

# Check pip version
pip --version

# List installed packages
pip list

# Check DVC version
dvc version

# Check kubectl version
kubectl version

# Check Minikube version
minikube version

# Check Docker version
docker --version

# Test network connectivity
ping google.com
curl https://api.github.com

# Check disk space
df -h

# Check memory
free -h

# Check processes
ps aux | grep python
ps aux | grep minikube
```

## Useful One-Liners

```bash
# Kill process on port 8080
# Windows
netstat -ano | findstr :8080
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8080 | xargs kill -9

# Find large files
du -sh * | sort -h

# Count lines of code
find . -name "*.py" | xargs wc -l

# Search in files
grep -r "search_term" .

# Create directory structure
mkdir -p data/{raw,processed} models components
```

## Quick Setup (All-in-One)

```bash
# Complete setup in one go
python -m venv venv && \
source venv/bin/activate && \
pip install -r requirements.txt && \
dvc init && \
python src/generate_dataset.py && \
dvc add data/raw_data.csv && \
git add . && \
git commit -m "Initial setup" && \
python pipeline.py
```

## Verification Commands

```bash
# Verify everything is working
echo "=== Python ===" && python --version && \
echo "=== Pip ===" && pip --version && \
echo "=== DVC ===" && dvc version && \
echo "=== Minikube ===" && minikube status && \
echo "=== Kubectl ===" && kubectl version --short && \
echo "=== Docker ===" && docker --version
```

## Cleanup Commands

```bash
# Clean Python cache
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Clean DVC cache
dvc gc

# Clean Docker
docker system prune -a

# Clean Minikube
minikube delete
rm -rf ~/.minikube

# Clean virtual environment
deactivate
rm -rf venv
```

## Screenshot Commands

```bash
# Get system info for screenshots
echo "System: $(uname -a)"
echo "Python: $(python --version)"
echo "Date: $(date)"

# Show directory tree (if tree is installed)
tree -L 2

# Or use ls
ls -la
```

---

## Tips

1. **Always activate virtual environment** before running Python commands
2. **Use `git status`** frequently to track changes
3. **Check `kubectl get pods -n kubeflow`** to monitor Kubeflow status
4. **Use `dvc status`** before pushing to ensure data is tracked
5. **Keep terminal logs** for debugging and screenshots

---

## Emergency Recovery

If something goes wrong:

```bash
# Reset DVC
rm -rf .dvc
dvc init

# Reset Minikube
minikube delete
minikube start

# Reset Git (careful!)
git reset --hard HEAD

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```
