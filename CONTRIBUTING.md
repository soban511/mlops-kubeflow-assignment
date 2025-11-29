# Contributing to MLOps Pipeline Project

## Development Setup

1. Fork the repository
2. Clone your fork
3. Create a virtual environment
4. Install dependencies: `pip install -r requirements.txt`
5. Create a feature branch

## Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Write meaningful commit messages

## Testing

Before submitting:
- Run syntax validation: `python -m py_compile src/*.py`
- Test pipeline execution: `python mlflow_pipeline.py`
- Verify CI passes: Check GitHub Actions

## Pull Request Process

1. Update documentation if needed
2. Ensure all tests pass
3. Update README.md with any new features
4. Request review from maintainers