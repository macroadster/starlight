# Production Deployment Checklist

**Version**: V4  
**Updated**: November 25, 2025  
**Required Completion**: 100% before production deployment  

---

## 🎯 Pre-Deployment Validation

### ✅ Environment Readiness

| Item | Status | Notes | Owner |
|------|--------|-------|-------|
| Kubernetes cluster v1.24+ available | ☐ | Verify nodes Ready | DevOps |
| 4+ CPU cores, 8GB+ RAM available | ☐ | Check resource allocation | DevOps |
| kubectl configured for target cluster | ☐ | Test `kubectl cluster-info` | Deployer |
| Helm 3 installed locally | ☐ | Version 3.8+ | Deployer |
| Docker registry access configured | ☐ | Test push/pull permissions | Deployer |
| Domain name configured (if using ingress) | ☐ | DNS records pointing | DevOps |

### ✅ Model Validation

| Item | Status | Notes | Owner |
|------|--------|-------|-------|
| `models/detector.onnx` exists and valid | ☐ | Run model validation script | ML Engineer |
| Model loads without errors | ☐ | Check ONNX runtime compatibility | ML Engineer |
| Inference produces valid output | ☐ | Test with sample input | ML Engineer |
| INT8 quantization completed | ☐ | FPR increase <0.001 | ML Engineer |
| Quantized model validated | ☐ | Performance within SLO | ML Engineer |

### ✅ Docker Image Validation

| Item | Status | Notes | Owner |
|------|--------|-------|-------|
| Dockerfile builds successfully | ☐ | No build errors | Deployer |
| Container starts without errors | ☐ | Check entrypoint script | Deployer |
| Model loads in container | ☐ | Verify volume mounts | Deployer |
| Health endpoint responds | ☐ | `/health` returns 200 | Deployer |
| Image pushed to registry | ☐ | Tag: `v4-prod` | Deployer |

### ✅ Performance Benchmarking

| Item | Status | Target | Actual | Notes |
|------|--------|--------|--------|-------|
| Inference latency (P95) | ☐ | <100ms | | Run benchmark script |
| Throughput (req/sec) | ☐ | >10 | | Load test results |
| Memory usage per pod | ☐ | <2GB | | Monitor during test |
| CPU usage per pod | ☐ | <1 core | | Monitor during test |
| Error rate | ☐ | <1% | | Load test validation |

---

## 🚀 Deployment Phase Checklist

### ✅ Infrastructure Deployment

| Item | Status | Command | Owner |
|------|--------|---------|-------|
| Namespace created | ☐ | `kubectl create ns starlight-prod` | DevOps |
| RBAC policies applied | ☐ | `kubectl apply -f k8s/rbac.yaml` | DevOps |
| PostgreSQL deployed | ☐ | `helm install starlight-db` | DevOps |
| Redis deployed | ☐ | `helm install starlight-redis` | DevOps |
| Database ready | ☐ | `kubectl wait pod -l app=postgresql` | DevOps |
| Cache ready | ☐ | `kubectl wait pod -l app=redis` | DevOps |

### ✅ Configuration Management

| Item | Status | Command | Owner |
|------|--------|---------|-------|
| ConfigMap created | ☐ | `kubectl create configmap` | Deployer |
| Secrets created | ☐ | `kubectl create secret generic` | Deployer |
| Environment variables set | ☐ | Verify in deployment | Deployer |
| Model volumes mounted | ☐ | Check pod spec | Deployer |
| Database connection configured | ☐ | Test connectivity | Deployer |

### ✅ Application Deployment

| Item | Status | Command | Owner |
|------|--------|---------|-------|
| Deployment applied | ☐ | `kubectl apply -f k8s/deployment.yaml` | Deployer |
| Service created | ☐ | `kubectl apply -f k8s/service.yaml` | Deployer |
| Ingress configured | ☐ | `kubectl apply -f k8s/ingress.yaml` | Deployer |
| Pods running | ☐ | `kubectl get pods -l app=starlight` | Deployer |
| Rollout successful | ☐ | `kubectl rollout status deployment` | Deployer |

### ✅ Health Checks

| Item | Status | Test | Expected | Owner |
|------|--------|------|----------|-------|
| Pod readiness | ☐ | `kubectl wait pod` | Ready condition | Deployer |
| Service health | ☐ | `curl /health` | 200 OK | Deployer |
| Model loading | ☐ | `curl /models/versions` | Returns model info | Deployer |
| Inference test | ☐ | `curl /inference` | Valid prediction | Deployer |
| Database connectivity | ☐ | Connection test | Success | Deployer |

---

## 🔄 Canary Deployment Checklist

### ✅ Canary Setup

| Item | Status | Command | Owner |
|------|--------|---------|-------|
| Canary deployment created | ☐ | `kubectl apply -f k8s/canary.yaml` | Deployer |
| Traffic split configured | ☐ | `kubectl apply -f k8s/traffic-split.yaml` | DevOps |
| 10% traffic to canary | ☐ | Verify in service mesh | DevOps |
| Canary pods healthy | ☐ | `kubectl get pods -l app=starlight-canary` | Deployer |
| Metrics collection active | ☐ | Check Prometheus | DevOps |

### ✅ Canary Monitoring (2-hour period)

| Time Check | Status | Latency | Error Rate | CPU | Memory | Notes |
|------------|--------|---------|------------|-----|--------|-------|
| T+15min | ☐ | | | | | |
| T+30min | ☐ | | | | | |
| T+60min | ☐ | | | | | |
| T+90min | ☐ | | | | | |
| T+120min | ☐ | | | | | Final check |

**Canary Success Criteria:**
- ✅ Latency <100ms (P95)
- ✅ Error rate <1%
- ✅ No pod restarts
- ✅ CPU <80%
- ✅ Memory <80%

---

## ✅ Production Migration Checklist

### ✅ Traffic Migration

| Item | Status | Command | Owner |
|------|--------|---------|-------|
| 100% traffic to V4 | ☐ | `kubectl patch service` | DevOps |
| Canary removed | ☐ | `kubectl delete deployment starlight-canary` | Deployer |
| All pods healthy | ☐ | `kubectl get pods -l app=starlight` | Deployer |
| Service endpoints responding | ☐ | `curl` tests | Deployer |
| Load balancer updated | ☐ | Check ingress controller | DevOps |

### ✅ Monitoring Verification

| Item | Status | Test | Expected | Owner |
|------|--------|------|----------|-------|
| Prometheus metrics | ☐ | Query metrics | Data flowing | DevOps |
| Grafana dashboards | ☐ | Access dashboards | Visualizing data | DevOps |
| Alert rules active | ☐ | Check alertmanager | Rules loaded | DevOps |
| Log aggregation | ☐ | Check logs | Centralized | DevOps |
| Health checks | ☐ | Continuous monitoring | All green | DevOps |

---

## 🛡️ Safety & Rollback Checklist

### ✅ Rollback Preparation

| Item | Status | Command | Owner |
|------|--------|---------|-------|
| Previous image available | ☐ | Check registry | `v3-stable` tag | DevOps |
| Rollback script tested | ☐ | `./scripts/deploy.sh rollback` | Success | Deployer |
| Database backup taken | ☐ | `pg_dump` | Backup stored | DevOps |
| Configuration saved | ☐ | Git tag current state | Version tagged | Deployer |
| Team notified | ☐ | Slack announcement | All aware | Tech Lead |

### ✅ Rollback Triggers

| Trigger | Threshold | Action | Owner |
|---------|-----------|--------|-------|
| High latency | >200ms for 5min | Auto-rollback | System |
| High error rate | >5% for 2min | Auto-rollback | System |
| Pod crashes | >3 restarts in 5min | Manual rollback | DevOps |
| Memory leak | >90% for 10min | Manual rollback | DevOps |
| Database errors | Connection failures | Manual rollback | DevOps |

---

## 📊 Post-Deployment Verification

### ✅ Functional Testing

| Item | Status | Test | Result | Owner |
|------|--------|------|--------|-------|
| Image inference | ☐ | Test various formats | Success | QA |
| Stego detection | ☐ | Test with known stego | Detected | QA |
| Clean image handling | ☐ | Test with clean images | Not detected | QA |
| API endpoints | ☐ | Test all endpoints | Working | QA |
| File size limits | ☐ | Test large files | Handled | QA |

### ✅ Performance Validation

| Metric | Target | Actual | Status | Owner |
|--------|--------|--------|--------|-------|
| P95 Latency | <100ms | | ☐ | DevOps |
| Throughput | >10 req/sec | | ☐ | DevOps |
| Availability | >99.9% | | ☐ | DevOps |
| FPR Rate | <0.07% | | ☐ | ML Engineer |
| Memory per pod | <2GB | | ☐ | DevOps |

### ✅ Security Validation

| Item | Status | Test | Result | Owner |
|------|--------|------|--------|-------|
| Authentication | ☐ | Test auth endpoints | Secured | Security |
| Input validation | ☐ | Test malicious inputs | Rejected | Security |
| Rate limiting | ☐ | Load test limits | Enforced | Security |
| TLS encryption | ☐ | Check HTTPS | Active | Security |
| Secrets management | ☐ | Verify no secrets in logs | Clean | Security |

---

## 📋 Documentation & Communication

### ✅ Documentation Updates

| Item | Status | Location | Owner |
|------|--------|----------|-------|
| Deployment guide updated | ☐ | `docs/PRODUCTION_DEPLOYMENT_GUIDE.md` | Tech Writer |
| Architecture documented | ☐ | `docs/V4_ARCHITECTURE_GUIDE.md` | Tech Writer |
| API specifications updated | ☐ | `docs/MONITORING_API_SPEC.md` | Tech Writer |
| Runbooks completed | ☐ | `docs/OPERATIONS_RUNBOOK.md` | Tech Writer |
| Troubleshooting guide | ☐ | `docs/TROUBLESHOOTING_GUIDE.md` | Tech Writer |

### ✅ Team Communication

| Item | Status | Audience | Channel | Owner |
|------|--------|----------|---------|-------|
| Deployment announcement | ☐ | All teams | Slack #announcements | Tech Lead |
| Performance summary | ☐ | Engineering | Email | DevOps |
| Customer notification | ☐ | Customers | Blog/Email | Product |
| Incident procedures | ☐ | Ops team | Training | DevOps |
| Success celebration | ☐ | All teams | Slack #general | Tech Lead |

---

## ✅ Final Sign-off

### 🎯 Critical Success Criteria

| Criteria | Status | Sign-off | Role |
|----------|--------|----------|------|
| Zero critical bugs | ☐ | | Tech Lead |
| Performance SLOs met | ☐ | | DevOps |
| Security scan passed | ☐ | | Security |
| All tests passing | ☐ | | QA |
| Documentation complete | ☐ | | Tech Writer |
| Team trained on procedures | ☐ | | Tech Lead |

### 📝 Final Approval

| Role | Name | Signature | Date | Status |
|------|------|-----------|------|--------|
| Tech Lead | | | | ☐ |
| DevOps Lead | | | | ☐ |
| ML Engineer | | | | ☐ |
| QA Engineer | | | | ☐ |
| Product Manager | | | | ☐ |

---

## 🚨 Emergency Contacts

| Role | Contact | Method | Response Time |
|------|---------|--------|---------------|
| On-call Engineer | [Name] | Phone/Slack | 15 min |
| Tech Lead | [Name] | Phone/Slack | 30 min |
| DevOps Lead | [Name] | Phone/Slack | 30 min |
| Incident Commander | [Name] | Phone/Slack | 15 min |

---

## 📝 Notes & Observations

```
[Deployment notes, issues encountered, lessons learned, etc.]
```

---

**Checklist Version**: 1.0  
**Last Updated**: November 25, 2025  
**Next Review**: After first production deployment  
**Maintainer**: GPT-OSS (Documentation & API Infrastructure)

---

## 🎯 Completion Instructions

1. **All items must be checked (☑) before production deployment**
2. **Any failed items must be resolved or documented with mitigation plan**
3. **Emergency rollback procedures must be tested and verified**
4. **All team members must sign-off in their respective areas**
5. **Post-deployment monitoring must be active for 24 hours**

**Remember**: If any critical item fails, do not proceed with deployment.