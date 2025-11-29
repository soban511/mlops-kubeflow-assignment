pipeline {
    agent any
    
    stages {
        stage('Environment Setup') {
            steps {
                echo 'Setting up environment...'
                sh '''
                    python3 --version
                    pip3 install -r requirements.txt
                '''
            }
        }
        
        stage('Pipeline Compilation') {
            steps {
                echo 'Compiling Kubeflow pipeline...'
                sh '''
                    python3 pipeline.py
                    if [ -f pipeline.yaml ]; then
                        echo "Pipeline compiled successfully!"
                    else
                        echo "Pipeline compilation failed!"
                        exit 1
                    fi
                '''
            }
        }
        
        stage('Validation') {
            steps {
                echo 'Validating pipeline components...'
                sh '''
                    python3 -c "import yaml; yaml.safe_load(open('pipeline.yaml'))"
                    echo "Pipeline YAML is valid!"
                '''
            }
        }
    }
    
    post {
        success {
            echo 'Pipeline build completed successfully!'
        }
        failure {
            echo 'Pipeline build failed!'
        }
    }
}
