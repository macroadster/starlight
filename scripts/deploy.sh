#!/bin/bash

# Starlight V4 Production Deployment Script
# Automated deployment with safety checks and rollback capability

set -euo pipefail

# Configuration
NAMESPACE="starlight-prod"
REGISTRY="${REGISTRY:-your-registry.com}"
IMAGE_TAG="${IMAGE_TAG:-v4-prod}"
CANARY_DURATION="${CANARY_DURATION:-7200}"  # 2 hours in seconds
DRY_RUN="${DRY_RUN:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Utility functions
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl access
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot access Kubernetes cluster"
        exit 1
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "Helm is not installed"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        log_warning "Namespace $NAMESPACE does not exist, will be created"
    fi
    
    log_success "Prerequisites check passed"
}

validate_model() {
    log_info "Validating ONNX model..."
    
    local model_path="models/detector.onnx"
    if [[ ! -f "$model_path" ]]; then
        log_error "Model file not found: $model_path"
        exit 1
    fi
    
    # Test model loading
    python3 -c "
import onnxruntime as ort
import numpy as np
try:
    session = ort.InferenceSession('$model_path')
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    outputs = session.run(None, {input_name: dummy_input})
    print('Model validation passed')
except Exception as e:
    print(f'Model validation failed: {e}')
    exit(1)
" || {
        log_error "Model validation failed"
        exit 1
    }
    
    log_success "Model validation passed"
}

build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
    local image_name="${REGISTRY}/starlight:${IMAGE_TAG}"
    
    # Build image
    log_info "Building image: $image_name"
    if [[ "$DRY_RUN" != "true" ]]; then
        docker build -t "$image_name" .
    fi
    
    # Test image
    log_info "Testing Docker image..."
    if [[ "$DRY_RUN" != "true" ]]; then
        docker run --rm -v $(pwd)/models:/app/models \
            "$image_name" python3 -c "
import onnxruntime as ort
session = ort.InferenceSession('/app/models/detector.onnx')
print('Container test passed')
"
    fi
    
    # Push image
    log_info "Pushing image to registry..."
    if [[ "$DRY_RUN" != "true" ]]; then
        docker push "$image_name"
    fi
    
    log_success "Image build and push completed: $image_name"
}

deploy_infrastructure() {
    log_info "Deploying infrastructure components..."
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        log_info "Creating namespace: $NAMESPACE"
        if [[ "$DRY_RUN" != "true" ]]; then
            kubectl create namespace $NAMESPACE
        fi
    fi
    
    # Deploy PostgreSQL
    log_info "Deploying PostgreSQL..."
    if [[ "$DRY_RUN" != "true" ]]; then
        helm repo add bitnami https://charts.bitnami.com/bitnami &> /dev/null || true
        helm upgrade --install starlight-db bitnami/postgresql \
            --namespace $NAMESPACE \
            --set auth.postgresPassword=secure_password \
            --set primary.persistence.size=10Gi \
            --wait
    fi
    
    # Deploy Redis
    log_info "Deploying Redis..."
    if [[ "$DRY_RUN" != "true" ]]; then
        helm upgrade --install starlight-redis bitnami/redis \
            --namespace $NAMESPACE \
            --set auth.password=secure_redis_password \
            --set master.persistence.size=2Gi \
            --wait
    fi
    
    log_success "Infrastructure deployment completed"
}

create_configurations() {
    log_info "Creating configurations and secrets..."
    
    # Create ConfigMap
    if [[ "$DRY_RUN" != "true" ]]; then
        kubectl create configmap starlight-config \
            --from-file=config/production.yaml \
            --namespace $NAMESPACE \
            --dry-run=client -o yaml | kubectl apply -f -
    fi
    
    # Create secrets
    if [[ "$DRY_RUN" != "true" ]]; then
        kubectl create secret generic starlight-secrets \
            --from-literal=db-password=secure_password \
            --from-literal=redis-password=secure_redis_password \
            --namespace $NAMESPACE \
            --dry-run=client -o yaml | kubectl apply -f -
    fi
    
    log_success "Configurations and secrets created"
}

deploy_application() {
    log_info "Deploying Starlight application..."
    
    local image_name="${REGISTRY}/starlight:${IMAGE_TAG}"
    
    # Update deployment with new image
    if [[ "$DRY_RUN" != "true" ]]; then
        # Apply deployment manifests
        kubectl apply -f k8s/rbac.yaml -n $NAMESPACE
        kubectl apply -f k8s/deployment.yaml -n $NAMESPACE
        kubectl apply -f k8s/service.yaml -n $NAMESPACE
        kubectl apply -f k8s/ingress.yaml -n $NAMESPACE
        
        # Update image in deployment
        kubectl set image deployment/starlight \
            starlight="$image_name" \
            -n $NAMESPACE
        
        # Wait for rollout
        kubectl rollout status deployment/starlight -n $NAMESPACE --timeout=600s
    fi
    
    log_success "Application deployment completed"
}

run_health_checks() {
    log_info "Running health checks..."
    
    # Wait for pods to be ready
    if [[ "$DRY_RUN" != "true" ]]; then
        kubectl wait --for=condition=ready pod -l app=starlight -n $NAMESPACE --timeout=300s
    fi
    
    # Test service health
    log_info "Testing service health endpoint..."
    if [[ "$DRY_RUN" != "true" ]]; then
        kubectl port-forward svc/starlight 8080:8080 -n $NAMESPACE &
        local port_forward_pid=$!
        
        sleep 5  # Give port-forward time to establish
        
        if curl -f http://localhost:8080/health &> /dev/null; then
            log_success "Health check passed"
        else
            log_error "Health check failed"
            kill $port_forward_pid 2>/dev/null || true
            exit 1
        fi
        
        kill $port_forward_pid 2>/dev/null || true
    fi
    
    log_success "All health checks passed"
}

deploy_canary() {
    log_info "Deploying canary for gradual rollout..."
    
    if [[ "$DRY_RUN" != "true" ]]; then
        # Create canary deployment
        kubectl apply -f k8s/canary.yaml -n $NAMESPACE
        
        # Configure traffic split (10% to canary)
        kubectl apply -f k8s/traffic-split.yaml -n $NAMESPACE
        
        log_info "Canary deployed. Monitoring for ${CANARY_DURATION} seconds..."
        
        # Monitor canary during the waiting period
        local end_time=$(($(date +%s) + CANARY_DURATION))
        while [[ $(date +%s) -lt $end_time ]]; do
            if [[ "$DRY_RUN" != "true" ]]; then
                local canary_pods=$(kubectl get pods -n $NAMESPACE -l app=starlight-canary --no-headers | wc -l)
                local ready_pods=$(kubectl get pods -n $NAMESPACE -l app=starlight-canary --no-headers | grep "Running" | wc -l)
                
                log_info "Canary status: $ready_pods/$canary_pods pods ready"
                
                if [[ $ready_pods -eq 0 ]]; then
                    log_error "Canary pods are not ready"
                    rollback_deployment
                    exit 1
                fi
            fi
            
            sleep 30  # Check every 30 seconds
        done
        
        log_success "Canary period completed successfully"
    fi
}

promote_to_production() {
    log_info "Promoting to full production..."
    
    if [[ "$DRY_RUN" != "true" ]]; then
        # Migrate 100% traffic to new version
        kubectl patch service starlight -n $NAMESPACE \
            -p '{"spec":{"selector":{"version":"v4"}}}'
        
        # Remove canary
        kubectl delete deployment starlight-canary -n $NAMESPACE --ignore-not-found=true
        
        # Final health check
        kubectl wait --for=condition=ready pod -l app=starlight -n $NAMESPACE --timeout=300s
        
        log_success "Promoted to full production"
    fi
}

rollback_deployment() {
    log_warning "Starting rollback procedure..."
    
    if [[ "$DRY_RUN" != "true" ]]; then
        # Scale down current deployment
        kubectl scale deployment starlight --replicas=0 -n $NAMESPACE
        
        # Restore previous version (assuming v3-stable is available)
        local previous_image="${REGISTRY}/starlight:v3-stable"
        kubectl set image deployment/starlight \
            starlight="$previous_image" \
            -n $NAMESPACE
        
        # Restore replicas
        kubectl scale deployment starlight --replicas=3 -n $NAMESPACE
        
        # Wait for rollout
        kubectl rollout status deployment/starlight -n $NAMESPACE --timeout=600s
        
        log_success "Rollback completed"
    fi
}

cleanup() {
    log_info "Performing cleanup..."
    
    # Remove any temporary port forwards
    pkill -f "kubectl port-forward.*starlight" 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Main execution flow
main() {
    log_info "Starting Starlight V4 production deployment..."
    log_info "Namespace: $NAMESPACE"
    log_info "Image: ${REGISTRY}/starlight:${IMAGE_TAG}"
    log_info "Dry run: $DRY_RUN"
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Execute deployment phases
    check_prerequisites
    validate_model
    build_and_push_image
    deploy_infrastructure
    create_configurations
    deploy_application
    run_health_checks
    deploy_canary
    promote_to_production
    
    log_success "🎉 Starlight V4 deployment completed successfully!"
    log_info "Service is now live in namespace: $NAMESPACE"
}

# Handle script arguments
case "${1:-}" in
    "validate")
        check_prerequisites
        validate_model
        ;;
    "build")
        build_and_push_image
        ;;
    "deploy")
        deploy_infrastructure
        create_configurations
        deploy_application
        ;;
    "health")
        run_health_checks
        ;;
    "rollback")
        rollback_deployment
        ;;
    "dry-run")
        DRY_RUN=true
        main
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  validate    - Validate prerequisites and model"
        echo "  build       - Build and push Docker image"
        echo "  deploy      - Deploy infrastructure and application"
        echo "  health      - Run health checks"
        echo "  rollback    - Rollback to previous version"
        echo "  dry-run     - Execute full deployment in dry-run mode"
        echo "  help        - Show this help message"
        echo ""
        echo "Environment variables:"
        echo "  REGISTRY    - Docker registry (default: your-registry.com)"
        echo "  IMAGE_TAG   - Image tag (default: v4-prod)"
        echo "  NAMESPACE   - Kubernetes namespace (default: starlight-prod)"
        echo "  DRY_RUN     - Enable dry-run mode (default: false)"
        ;;
    "")
        main
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac