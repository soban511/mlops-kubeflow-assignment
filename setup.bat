@echo off
REM MLOps Kubeflow Assignment Setup Script for Windows

echo =========================================
echo MLOps Kubeflow Assignment Setup
echo =========================================

REM Check Python version
echo Checking Python version...
python --version

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing Python dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Initialize DVC
echo Initializing DVC...
dvc init

REM Create directory structure
echo Creating directory structure...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "models" mkdir models
if not exist "components" mkdir components

echo =========================================
echo Setup completed successfully!
echo =========================================
echo.
echo Next steps:
echo 1. Activate virtual environment: venv\Scripts\activate
echo 2. Configure DVC remote: dvc remote add -d myremote ^<path^>
echo 3. Add dataset: dvc add data\raw_data.csv
echo 4. Compile pipeline: python pipeline.py
echo.
pause
