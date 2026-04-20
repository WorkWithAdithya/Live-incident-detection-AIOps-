// Jenkinsfile
// ─────────────────────────────────────────────────────────────────────────────
// AIOps Incident Detection — Jenkins CI/CD Pipeline
//
// Pipeline stages:
//   1. Checkout       → clone repo
//   2. Lint           → Python (flake8) + JS (eslint)
//   3. Test           → Python unit tests (pytest)
//   4. Build Images   → docker-compose build all 3 services
//   5. Health Check   → start containers, verify all pass healthcheck
//   6. Deploy         → docker-compose up -d (replace running containers)
//   7. Notify         → print summary (extend to Slack/email if needed)
//
// Prerequisites (install on Jenkins agent):
//   - Docker + Docker Compose
//   - Python 3.11+ with pip
//   - Node.js 20+
//
// Setup in Jenkins:
//   1. New Item → Pipeline
//   2. Definition: Pipeline script from SCM
//   3. SCM: Git → your repo URL
//   4. Script Path: Jenkinsfile
//   5. Add credentials for .env if needed (see CREDENTIALS section)

pipeline {

    agent any

    // ── Global options ────────────────────────────────────────────────────────
    options {
        timeout(time: 30, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
        disableConcurrentBuilds()
        timestamps()
    }

    // ── Environment ───────────────────────────────────────────────────────────
    environment {
        COMPOSE_FILE      = 'docker-compose.yml'
        COMPOSE_PROJECT   = 'aiops'
        PYTHON_VERSION    = '3.11'

        // Docker image tags — use git short SHA for traceability
        GIT_SHA           = sh(script: 'git rev-parse --short HEAD',
                               returnStdout: true).trim()
        BUILD_TAG         = "${env.BUILD_NUMBER}-${env.GIT_SHA}"

        // Paths
        AI_MODEL_DIR      = 'ai_model'
        BACKEND_DIR       = 'frontend/backend'
        FRONTEND_DIR      = 'frontend/ui'
        LOG_GEN_DIR       = 'log_generator'
    }

    // ── Stages ────────────────────────────────────────────────────────────────
    stages {

        // ── Stage 1: Checkout ─────────────────────────────────────────────────
        stage('Checkout') {
            steps {
                echo "📥 Checking out commit ${env.GIT_SHA}"
                checkout scm

                // Verify critical files exist
                sh '''
                    echo "Verifying project structure..."
                    test -f log_generator/Dockerfile      || (echo "❌ Missing log_generator/Dockerfile" && exit 1)
                    test -f frontend/backend/Dockerfile   || (echo "❌ Missing frontend/backend/Dockerfile" && exit 1)
                    test -f frontend/ui/Dockerfile        || (echo "❌ Missing frontend/ui/Dockerfile" && exit 1)
                    test -f docker-compose.yml            || (echo "❌ Missing docker-compose.yml" && exit 1)
                    test -f log_generator/.env            || (echo "❌ Missing log_generator/.env — add DATABASE_URL" && exit 1)
                    echo "✅ Project structure OK"
                '''
            }
        }

        // ── Stage 2: Lint ─────────────────────────────────────────────────────
        stage('Lint') {
            parallel {

                stage('Python Lint') {
                    steps {
                        echo "🔍 Linting Python..."
                        sh '''
                            python3 -m pip install --quiet flake8
                            # Lint backend
                            flake8 frontend/backend/ \
                                --max-line-length=120 \
                                --exclude=__pycache__,venv,.venv \
                                --count --statistics \
                                || echo "⚠️  Backend lint warnings (non-blocking)"

                            # Lint ai_model
                            flake8 ai_model/model/ \
                                --max-line-length=120 \
                                --exclude=__pycache__,venv,.venv \
                                --count --statistics \
                                || echo "⚠️  AI model lint warnings (non-blocking)"

                            # Lint log_generator
                            flake8 log_generator/src/ \
                                --max-line-length=120 \
                                --exclude=__pycache__,venv,.venv \
                                --count --statistics \
                                || echo "⚠️  Log generator lint warnings (non-blocking)"
                        '''
                    }
                }

                stage('JS Lint') {
                    steps {
                        echo "🔍 Linting JavaScript/React..."
                        dir('frontend/ui') {
                            sh '''
                                npm install --silent
                                # Run eslint if config exists
                                if [ -f .eslintrc.js ] || [ -f .eslintrc.json ] || [ -f eslint.config.js ]; then
                                    npx eslint src/ --ext .js,.jsx \
                                        --max-warnings 20 \
                                        || echo "⚠️  JS lint warnings (non-blocking)"
                                else
                                    echo "ℹ️  No ESLint config found — skipping JS lint"
                                fi
                            '''
                        }
                    }
                }
            }
        }

        // ── Stage 3: Test ─────────────────────────────────────────────────────
        stage('Test') {
            steps {
                echo "🧪 Running Python tests..."
                sh '''
                    python3 -m pip install --quiet pytest torch numpy scikit-learn pandas

                    # Run tests if they exist
                    if [ -d tests/ ]; then
                        pytest tests/ -v \
                            --tb=short \
                            --junit-xml=test-results.xml \
                            || exit 1
                    else
                        echo "ℹ️  No tests/ directory found — running model sanity checks"
                    fi

                    # Sanity check: LSTM Autoencoder instantiates correctly
                    python3 - << 'PYEOF'
import sys
sys.path.insert(0, "ai_model")
import torch
from model.lstm_autoencoder import LSTMAutoencoder
from model.lstm_forecaster  import LSTMForecaster

ae = LSTMAutoencoder()
fc = LSTMForecaster()

x = torch.randn(2, 60, 3)
out_ae = ae(x)
assert out_ae.shape == (2, 60, 3), f"Autoencoder output shape wrong: {out_ae.shape}"

out_fc = fc.predict(x)
assert out_fc.shape == (2, 12, 3), f"Forecaster output shape wrong: {out_fc.shape}"

print("✅ Model sanity checks passed")
print(f"   Autoencoder output : {out_ae.shape}")
print(f"   Forecaster output  : {out_fc.shape}")
PYEOF
                '''
            }
            post {
                always {
                    // Publish test results if they exist
                    script {
                        if (fileExists('test-results.xml')) {
                            junit 'test-results.xml'
                        }
                    }
                }
            }
        }

        // ── Stage 4: Build Docker Images ──────────────────────────────────────
        stage('Build Images') {
            steps {
                echo "🐳 Building Docker images (tag: ${env.BUILD_TAG})..."
                sh '''
                    docker-compose -f docker-compose.yml -p ${COMPOSE_PROJECT} \
                        build \
                        --no-cache \
                        --parallel \
                        2>&1 | tee build.log

                    echo "✅ All images built successfully"
                    docker images | grep aiops || true
                '''
            }
            post {
                failure {
                    sh 'cat build.log || true'
                }
            }
        }

        // ── Stage 5: Health Check ─────────────────────────────────────────────
        stage('Health Check') {
            steps {
                echo "🏥 Starting containers and verifying health..."
                sh '''
                    # Bring up backend only first (no log_generator — needs DB)
                    docker-compose -f docker-compose.yml -p ${COMPOSE_PROJECT} \
                        up -d backend

                    # Wait for backend to be healthy (max 60s)
                    echo "Waiting for backend health..."
                    for i in $(seq 1 12); do
                        STATUS=$(docker inspect --format="{{.State.Health.Status}}" \
                                 aiops_backend 2>/dev/null || echo "missing")
                        echo "  Attempt ${i}/12: backend status = ${STATUS}"
                        if [ "${STATUS}" = "healthy" ]; then
                            echo "✅ Backend is healthy"
                            break
                        fi
                        if [ $i -eq 12 ]; then
                            echo "❌ Backend failed health check"
                            docker-compose -f docker-compose.yml -p ${COMPOSE_PROJECT} logs backend
                            exit 1
                        fi
                        sleep 5
                    done

                    # Test API endpoint
                    curl -sf http://localhost:8000/ | python3 -m json.tool
                    echo "✅ Backend API responding"

                    # Bring up frontend
                    docker-compose -f docker-compose.yml -p ${COMPOSE_PROJECT} \
                        up -d frontend

                    sleep 15
                    curl -sf http://localhost:5173 > /dev/null \
                        && echo "✅ Frontend responding" \
                        || echo "⚠️  Frontend not yet ready (may need more time)"
                '''
            }
            post {
                failure {
                    sh '''
                        echo "=== Backend logs ==="
                        docker-compose -f docker-compose.yml -p ${COMPOSE_PROJECT} logs backend || true
                        echo "=== Frontend logs ==="
                        docker-compose -f docker-compose.yml -p ${COMPOSE_PROJECT} logs frontend || true
                    '''
                }
            }
        }

        // ── Stage 6: Deploy ───────────────────────────────────────────────────
        stage('Deploy') {
            steps {
                echo "🚀 Deploying all services..."
                sh '''
                    docker-compose -f docker-compose.yml -p ${COMPOSE_PROJECT} \
                        up -d \
                        --remove-orphans

                    echo ""
                    echo "✅ Deployment complete!"
                    echo "   Frontend  : http://localhost:5173"
                    echo "   Backend   : http://localhost:8000"
                    echo "   API docs  : http://localhost:8000/docs"
                    echo ""
                    docker-compose -f docker-compose.yml -p ${COMPOSE_PROJECT} ps
                '''
            }
        }
    }

    // ── Post actions ──────────────────────────────────────────────────────────
    post {

        success {
            echo """
╔══════════════════════════════════════════════════════╗
║  ✅  BUILD #${env.BUILD_NUMBER} SUCCEEDED             ║
║  Commit : ${env.GIT_SHA}                             ║
║  Branch : ${env.GIT_BRANCH ?: 'unknown'}             ║
╚══════════════════════════════════════════════════════╝
Frontend  → http://localhost:5173
Backend   → http://localhost:8000
API docs  → http://localhost:8000/docs
"""
        }

        failure {
            echo """
╔══════════════════════════════════════════════════════╗
║  ❌  BUILD #${env.BUILD_NUMBER} FAILED               ║
║  Commit : ${env.GIT_SHA}                             ║
╚══════════════════════════════════════════════════════╝
Check the logs above for details.
"""
            // Stop any partially started containers on failure
            sh '''
                docker-compose -f docker-compose.yml -p ${COMPOSE_PROJECT} \
                    down --remove-orphans 2>/dev/null || true
            '''
        }

        always {
            echo "Build ${env.BUILD_NUMBER} finished — cleaning workspace..."
            cleanWs(
                cleanWhenSuccess:  false,   // keep workspace on success for inspection
                cleanWhenFailure:  true,
                cleanWhenAborted:  true,
                deleteDirs:        true,
                disableDeferredWipeout: true,
                notFailBuild:      true,
                patterns: [[pattern: 'build.log', type: 'INCLUDE'],
                           [pattern: '**/__pycache__', type: 'INCLUDE'],
                           [pattern: '**/node_modules', type: 'INCLUDE']]
            )
        }
    }
}