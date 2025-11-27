# Production Deployment Guide

**Version**: V4  
**Updated**: November 25, 2025  
**Target Deployment Time**: <2 hours  

---

## 🎯 Overview

This guide provides a complete, step-by-step playbook for deploying Starlight V4 to production environments. The deployment is designed to be safe, reversible, and observable at every stage.

### Prerequisites

- **Kubernetes cluster** (v1.24+) with 4+ CPU cores, 8GB+ RAM
- **Docker registry** access (Docker Hub, GCR, or private)
- **kubectl** configured for target cluster
- **Helm 3** installed locally
- **Domain name** for service endpoints (optional)

### Deployment Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   Ingress       │────│   Starlight     │
│   (External)    │    │   Controller    │    │   Service       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐            │
                       │   Monitoring    │────────────┤
                       │   API           │            │
                       └─────────────────┘            │
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Model Store   │────│   Redis Cache   │────│   PostgreSQL    │
│   (ONNX files)  │    │   (Metrics)     │    │   (Metadata)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 📋 Phase 1: Pre-Deployment (30 minutes)

### Step 1.1: Environment Validation

```bash
# Verify cluster access
kubectl cluster-info
kubectl get nodes

# Check resource availability
kubectl describe nodes | grep -A 5 "Allocated resources"

# Verify storage class
kubectl get storageclass
```

**Expected Output**: All nodes Ready, sufficient CPU/memory available

### Step 1.2: ONNX Model Validation

```bash
# Verify model files exist and are valid
python3 -c "
import onnxruntime as ort
import numpy as np

# Load the V4 model
model_path = 'models/detector.onnx'
session = ort.InferenceSession(model_path)

# Test inference
input_name = session.get_inputs()[0].name
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {input_name: dummy_input})

print(f'Model loaded successfully')
print(f'Input shape: {dummy_input.shape}')
print(f'Output shape: {outputs[0].shape}')
"
```

**Success Criteria**: Model loads without errors, inference produces valid output

### Step 1.3: INT8 Quantization + FPR Verification

```bash
# Run quantization script
python3 scripts/quantize_model.py \
  --input models/detector.onnx \
  --output models/detector_int8.onnx \
  --calibration-data data/validation/

# Verify FPR remains within acceptable range
python3 scripts/validate_quantized.py \
  --original models/detector.onnx \
  --quantized models/detector_int8.onnx \
  --threshold 0.001
```

**Success Criteria**: FPR increase <0.001, model size reduced by ~4x

### Step 1.4: Docker Build & Test

```bash
# Build production image
docker build -t starlight:v4-prod .

# Run container tests
docker run --rm -v $(pwd)/models:/app/models \
  starlight:v4-prod python3 -c "
import onnxruntime as ort
session = ort.InferenceSession('/app/models/detector.onnx')
print('Container test passed')
"

# Push to registry
docker tag starlight:v4-prod your-registry/starlight:v4-prod
docker push your-registry/starlight:v4-prod
```

**Success Criteria**: Image builds, tests pass, push succeeds

### Step 1.5: Performance Benchmarking

```bash
# Run baseline performance tests
python3 scripts/benchmark.py \
  --model models/detector.onnx \
  --iterations 1000 \
  --output benchmark_results.json

# Verify SLO compliance
python3 scripts/verify_slo.py \
  --benchmark benchmark_results.json \
  --max-latency 100 \
  --min-throughput 10
```

**Success Criteria**: Latency <100ms, Throughput >10 req/sec

---

## 🚀 Phase 2: Deployment (60 minutes)

### Step 2.1: Namespace & RBAC Setup

```bash
# Create namespace
kubectl create namespace starlight-prod

# Apply RBAC policies
kubectl apply -f k8s/rbac.yaml -n starlight-prod

# Verify service account
kubectl get serviceaccount -n starlight-prod
```

### Step 2.2: Database & Cache Deployment

```bash
# Deploy PostgreSQL
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install starlight-db bitnami/postgresql \
  --namespace starlight-prod \
  --set auth.postgresPassword=secure_password \
  --set primary.persistence.size=10Gi

# Deploy Redis
helm install starlight-redis bitnami/redis \
  --namespace starlight-prod \
  --set auth.password=secure_redis_password \
  --set master.persistence.size=2Gi

# Wait for readiness
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=postgresql -n starlight-prod --timeout=300s
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=redis -n starlight-prod --timeout=300s
```

### Step 2.3: Configuration Management

```bash
# Create ConfigMap with model settings
kubectl create configmap starlight-config \
  --from-file=config/production.yaml \
  --namespace starlight-prod

# Create secrets for sensitive data
kubectl create secret generic starlight-secrets \
  --from-literal=db-password=secure_password \
  --from-literal=redis-password=secure_redis_password \
  --namespace starlight-prod

# Verify configuration
kubectl get configmap starlight-config -n starlight-prod -o yaml
kubectl get secret starlight-secrets -n starlight-prod
```

### Step 2.4: Main Application Deployment

```bash
# Deploy Starlight service
kubectl apply -f k8s/deployment.yaml -n starlight-prod
kubectl apply -f k8s/service.yaml -n starlight-prod
kubectl apply -f k8s/ingress.yaml -n starlight-prod

# Monitor rollout progress
kubectl rollout status deployment/starlight -n starlight-prod --timeout=600s

# Check pod health
kubectl get pods -n starlight-prod
kubectl logs -l app=starlight -n starlight-prod --tail=50
```

### Step 2.5: Health Checks

```bash
# Test service health
kubectl port-forward svc/starlight 8080:8080 -n starlight-prod &
curl http://localhost:8080/health

# Verify model loading
curl http://localhost:8080/models/versions

# Test inference endpoint
curl -X POST http://localhost:8080/inference \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/test.png"}'

# Cleanup port forward
pkill -f "kubectl port-forward"
```

### Step 2.6: Canary Rollout (10% traffic for 2 hours)

```bash
# Create canary deployment
kubectl apply -f k8s/canary.yaml -n starlight-prod

# Configure Istio/NGINX for 10% traffic split
kubectl apply -f k8s/traffic-split.yaml -n starlight-prod

# Monitor canary metrics
kubectl top pods -n starlight-prod -l app=starlight-canary
kubectl logs -l app=starlight-canary -n starlight-prod --tail=20

# Wait 2 hours for stability
echo "Waiting 2 hours for canary validation..."
sleep 7200
```

---

## ✅ Phase 3: Verification (30 minutes)

### Step 3.1: Monitoring Metrics Validation

```bash
# Check Prometheus metrics
kubectl port-forward svc/prometheus 9090:9090 -n monitoring &
curl "http://localhost:9090/api/v1/query?query=starlight_inference_duration_seconds"

# Verify alert rules
curl "http://localhost:9090/api/v1/rules" | jq '.data.groups[].rules[] | select(.name=="StarlightHighLatency")'

# Check Grafana dashboards
kubectl port-forward svc/grafana 3000:3000 -n monitoring &
# Access http://localhost:3000/d/starlight-overview
```

### Step 3.2: Alert System Testing

```bash
# Trigger test alert
curl -X POST http://starlight-alerts.test/alert \
  -H "Content-Type: application/json" \
  -d '{
    "alertname": "StarlightTestAlert",
    "severity": "warning",
    "message": "Test alert for verification"
  }'

# Verify alert delivery
kubectl logs -l app=alertmanager -n monitoring --tail=10
```

### Step 3.3: Performance SLO Verification

```bash
# Run load test
python3 scripts/load_test.py \
  --endpoint http://starlight-prod.example.com/inference \
  --duration 300 \
  --concurrency 20 \
  --output load_test_results.json

# Verify SLO compliance
python3 scripts/verify_slo.py \
  --benchmark load_test_results.json \
  --max-latency 100 \
  --min-throughput 10 \
  --max-error-rate 0.01
```

### Step 3.4: Full Traffic Migration

```bash
# Migrate 100% traffic to new version
kubectl patch service starlight -n starlight-prod \
  -p '{"spec":{"selector":{"version":"v4"}}}'

# Remove canary
kubectl delete deployment starlight-canary -n starlight-prod

# Final health check
kubectl get pods -n starlight-prod -l app=starlight
kubectl rollout status deployment/starlight -n starlight-prod
```

---

## 🔄 Phase 4: Rollback Procedure

### Automated Rollback Script

```bash
#!/bin/bash
# scripts/rollback.sh

set -e

NAMESPACE="starlight-prod"
PREVIOUS_VERSION="v3-stable"

echo "Starting rollback to $PREVIOUS_VERSION..."

# Step 1: Scale down current deployment
kubectl scale deployment starlight --replicas=0 -n $NAMESPACE

# Step 2: Deploy previous version
kubectl set image deployment/starlight \
  starlight=your-registry/starlight:$PREVIOUS_VERSION \
  -n $NAMESPACE

# Step 3: Restore replicas
kubectl scale deployment starlight --replicas=3 -n $NAMESPACE

# Step 4: Wait for rollout
kubectl rollout status deployment/starlight -n $NAMESPACE --timeout=600s

# Step 5: Health check
kubectl wait --for=condition=ready pod -l app=starlight -n $NAMESPACE --timeout=300s

echo "Rollback completed successfully"
```

### Verification Post-Rollback

```bash
# Test service functionality
./scripts/health_check.sh --namespace starlight-prod

# Verify metrics collection
./scripts/verify_metrics.sh --namespace starlight-prod

# Document incident
cat > docs/incidents/rollback_$(date +%Y%m%d_%H%M%S).md << EOF
# Rollback Incident

**Time**: $(date)
**Reason**: [Specify reason for rollback]
**Version**: Rolled back from v4 to v3-stable
**Duration**: [Total rollback time]
**Impact**: [User impact description]
**Resolution**: [What was fixed before redeployment]

## Lessons Learned

[What went wrong and how to prevent it]
EOF
```

---

## 📊 Success Metrics

### Deployment Success Criteria

- ✅ All health checks passing
- ✅ Zero error logs in application
- ✅ Performance SLOs met (latency <100ms, throughput >10 req/sec)
- ✅ Monitoring metrics flowing correctly
- ✅ Alert system functional
- ✅ FPR <0.07% maintained

### Operational Readiness

- ✅ Documentation complete and accessible
- ✅ Runbooks tested and verified
- ✅ Team trained on procedures
- ✅ Monitoring dashboards configured
- ✅ Incident response process tested

---

## 🆘 Troubleshooting Quick Reference

| Issue | Symptom | Quick Fix |
|-------|---------|-----------|
| Pod CrashLoopBackOff | Container restarting | `kubectl logs pod-name` - check config/secrets |
| High Latency | >100ms response time | Check resource limits, scale horizontally |
| 503 Errors | Service unavailable | Verify ingress, check pod readiness |
| Model Loading Failed | ONNX load error | Validate model file, check storage access |
| Database Connection Failed | DB errors | Check credentials, verify network policies |

---

## 📞 Support & Escalation

### Primary Contacts
- **On-call Engineer**: [Contact info]
- **Tech Lead**: [Contact info]
- **Product Manager**: [Contact info]

### Escalation Path
1. **Level 1**: On-call engineer (15 min response)
2. **Level 2**: Tech lead (30 min response)
3. **Level 3**: Engineering manager (1 hour response)

### Communication Channels
- **Incidents**: `#starlight-incidents` Slack
- **Deployments**: `#starlight-deployments` Slack
- **Documentation**: Confluence space

---

**Last Updated**: November 25, 2025  
**Next Review**: After first production deployment  
**Maintainer**: GPT-OSS (Documentation & API Infrastructure)