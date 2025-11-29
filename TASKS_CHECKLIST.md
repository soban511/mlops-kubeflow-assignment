# MLOps Assignment Tasks Checklist

Use this checklist to track your progress through the assignment.

## Task 1: Project Initialization and Data Versioning (20 Marks)

### Setup (Completed ✓)
- [x] Create GitHub repository `mlops-kubeflow-assignment`
- [x] Clone repository locally
- [x] Create project structure:
  - [x] `data/` directory
  - [x] `src/` directory
  - [x] `components/` directory
  - [x] `pipeline_components.py`
  - [x] `model_training.py`
  - [x] `pipeline.py`
  - [x] `requirements.txt`
  - [x] `Dockerfile`
  - [x] `Jenkinsfile`
  - [x] `.gitignore`

### DVC Setup (To Do)
- [ ] Install DVC: `pip install dvc`
- [ ] Initialize DVC: `dvc init`
- [ ] Configure DVC remote storage
  - [ ] Option 1: Local folder: `dvc remote add -d myremote /path/to/storage`
  - [ ] Option 2: Google Drive: `dvc remote add -d myremote gdrive://folder_id`
  - [ ] Option 3: AWS S3: `dvc remote add -d myremote s3://bucket/path`
- [ ] Generate dataset: `python src/generate_dataset.py`
- [ ] Track dataset with DVC: `dvc add data/raw_data.csv`
- [ ] Commit DVC files: `git add data/raw_data.csv.dvc data/.gitignore .dvc/`
- [ ] Push data to remote: `dvc push`
- [ ] Verify: `dvc status`

### Deliverable 1 Screenshots
- [ ] Screenshot: GitHub repository file structure
- [ ] Screenshot: `dvc status` command output
- [ ] Screenshot: `dvc push` command output
- [ ] Screenshot: `requirements.txt` content

---

## Task 2: Building Kubeflow Pipeline Components (25 Marks)

### Component Development (Completed ✓)
- [x] Create `src/pipeline_components.py`
- [x] Implement Data Extraction component
  - [x] Use `@component` decorator
  - [x] Define inputs and outputs
  - [x] Load Boston Housing dataset
- [x] Implement Data Preprocessing component
  - [x] Clean and scale data
  - [x] Split into train/test sets
  - [x] Save processed datasets
- [x] Implement Model Training component
  - [x] Train Random Forest model
  - [x] Save model artifact
- [x] Implement Model Evaluation component
  - [x] Load model and test data
  - [x] Calculate metrics (RMSE, R2, accuracy)
  - [x] Save metrics to file

### Component Compilation (To Do)
- [ ] Test components locally
- [ ] Compile components to YAML (if needed)
- [ ] Verify component definitions

### Deliverable 2 Screenshots
- [ ] Screenshot: `src/pipeline_components.py` showing 2+ components
- [ ] Screenshot: `components/` directory with YAML files (if applicable)
- [ ] Document: Explanation of training component inputs/outputs

---

## Task 3: Orchestrating the Pipeline on Minikube (30 Marks)

### Minikube Setup (To Do)
- [ ] Install Minikube
- [ ] Start Minikube: `minikube start --cpus 4 --memory 8192`
- [ ] Verify cluster: `minikube status`
- [ ] Check nodes: `kubectl get nodes`

### Kubeflow Pipelines Installation (To Do)
- [ ] Set pipeline version: `export PIPELINE_VERSION=2.0.0`
- [ ] Install cluster resources:
  ```bash
  kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
  ```
- [ ] Wait for CRDs: `kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io`
- [ ] Install Kubeflow Pipelines:
  ```bash
  kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$PIPELINE_VERSION"
  ```
- [ ] Wait for pods: `kubectl wait --for=condition=ready --timeout=300s pods --all -n kubeflow`
- [ ] Port forward: `kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80`
- [ ] Access UI: http://localhost:8080

### Pipeline Definition and Execution (To Do)
- [ ] Review `pipeline.py`
- [ ] Compile pipeline: `python pipeline.py`
- [ ] Verify `pipeline.yaml` created
- [ ] Upload pipeline to KFP UI
- [ ] Create and start a run
- [ ] Monitor execution
- [ ] Verify all steps complete successfully

### Deliverable 3 Screenshots
- [ ] Screenshot: `minikube status` showing running cluster
- [ ] Screenshot: Kubeflow Pipelines UI with pipeline graph
- [ ] Screenshot: All pipeline steps connected and completed
- [ ] Screenshot: Pipeline run details showing model accuracy

---

## Task 4: Continuous Integration with Jenkins/GitHub Workflows (15 Marks)

### Jenkins Setup (Option 1)
- [ ] Install Jenkins
- [ ] Start Jenkins service
- [ ] Access Jenkins UI: http://localhost:8080
- [ ] Install required plugins (Git, Pipeline)
- [ ] Create new Pipeline job
- [ ] Configure job:
  - [ ] Link to GitHub repository
  - [ ] Set script path to `Jenkinsfile`
  - [ ] Configure webhook (optional)
- [ ] Trigger build manually
- [ ] Verify all stages pass:
  - [ ] Environment Setup
  - [ ] Pipeline Compilation
  - [ ] Validation

### GitHub Actions Setup (Option 2)
- [ ] Review `.github/workflows/pipeline.yml`
- [ ] Push code to GitHub
- [ ] Navigate to Actions tab
- [ ] Verify workflow runs automatically
- [ ] Check all jobs pass

### Deliverable 4 Screenshots
- [ ] Screenshot: Jenkins pipeline console output (all stages successful)
- [ ] Screenshot: `Jenkinsfile` content
- [ ] OR Screenshot: GitHub Actions workflow run (if using GitHub Actions)

---

## Task 5: Final Integration and Documentation (10 Marks)

### Documentation (Completed ✓)
- [x] Create comprehensive `README.md`
  - [x] Project Overview
  - [x] Setup Instructions
  - [x] Pipeline Walkthrough
  - [x] Troubleshooting guide
- [x] Create `QUICKSTART.md`
- [x] Create `TASKS_CHECKLIST.md`

### Final Steps (To Do)
- [ ] Review all code
- [ ] Test complete workflow end-to-end
- [ ] Ensure all files are committed
- [ ] Push to GitHub: `git push origin main`
- [ ] Verify repository is public
- [ ] Test cloning and setup from scratch

### Deliverable 5 Screenshots
- [ ] Screenshot: GitHub repository main page
- [ ] Screenshot: README.md visible in repository
- [ ] Screenshot: Complete project structure
- [ ] Document: GitHub repository URL

---

## Additional Recommendations

### Testing
- [ ] Test local training: `python src/model_training.py`
- [ ] Test pipeline compilation: `python pipeline.py`
- [ ] Test DVC pull on fresh clone
- [ ] Test complete setup from QUICKSTART.md

### Documentation
- [ ] Add comments to all Python functions
- [ ] Document hyperparameters
- [ ] Add troubleshooting tips
- [ ] Include performance metrics

### Best Practices
- [ ] Use meaningful commit messages
- [ ] Keep code clean and organized
- [ ] Follow PEP 8 style guide
- [ ] Add error handling
- [ ] Include logging statements

---

## Submission Checklist

Before submitting, ensure you have:

- [ ] All 5 tasks completed
- [ ] All required screenshots captured
- [ ] GitHub repository is public
- [ ] README.md is comprehensive
- [ ] All code is pushed to GitHub
- [ ] Repository URL is ready to submit
- [ ] All deliverables are documented

---

## Grading Breakdown

- **Task 1**: 20 marks - Project setup and DVC
- **Task 2**: 25 marks - Kubeflow components
- **Task 3**: 30 marks - Pipeline orchestration
- **Task 4**: 15 marks - CI/CD with Jenkins
- **Task 5**: 10 marks - Documentation
- **Total**: 100 marks

---

## Timeline Suggestion

- **Day 1**: Tasks 1 & 2 (Setup, DVC, Components)
- **Day 2**: Task 3 (Minikube, Kubeflow, Pipeline)
- **Day 3**: Task 4 & 5 (Jenkins, Documentation, Testing)

Good luck! 🚀
