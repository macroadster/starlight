# Starlight Bitcoin Scanning API Deployment Guide

## Overview

This guide covers deployment of the Starlight Bitcoin Scanning API, which provides REST endpoints for scanning Bitcoin transactions and images for steganography using the existing Starlight scanner.

## Prerequisites

### System Requirements
- **CPU**: 4+ cores recommended
- **Memory**: 8GB+ RAM
- **Storage**: 50GB+ available space
- **Network**: Stable internet connection for Bitcoin API access

### Software Dependencies
- Python 3.9+
- Docker (optional, for containerized deployment)
- Bitcoin Core node (optional, can use public APIs)

## Installation

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd starlight

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r bitcoin_api_requirements.txt
```

### 2. Model Setup

```bash
# Ensure the Starlight model is available
ls models/detector_balanced.onnx

# If not present, train a model first:
python3 trainer.py
```

### 3. Configuration

Create environment file `.env`:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
API_WORKERS=4

# Bitcoin Configuration
BITCOIN_NODE_URL=https://blockstream.info/api
BITCOIN_NODE_TIMEOUT=30

# Scanner Configuration
MODEL_PATH=models/detector_balanced.onnx
SCANNER_WORKERS=4
CONFIDENCE_THRESHOLD=0.5

# Security
API_KEY_SECRET=your-secret-key-here
JWT_SECRET=your-jwt-secret-here

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

## Deployment Options

### Option 1: Direct Python Deployment

```bash
# Run the API server
python3 bitcoin_api.py

# Or with uvicorn directly
uvicorn bitcoin_api:app --host 0.0.0.0 --port 8080 --workers 4
```

### Option 2: Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# Copy requirements and install Python packages
COPY bitcoin_api_requirements.txt .
RUN pip install --no-cache-dir -r bitcoin_api_requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 starlight
USER starlight

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start the application
CMD ["uvicorn", "bitcoin_api:app", "--host", "0.0.0.0", "--port", "8080"]
```

Build and run:

```bash
# Build image
docker build -t starlight-bitcoin-api .

# Run container
docker run -d \
    --name starlight-api \
    -p 8080:8080 \
    -v $(pwd)/models:/app/models \
    -e API_KEY_SECRET=your-secret-key \
    starlight-bitcoin-api
```

### Option 3: Kubernetes Deployment

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: starlight-bitcoin-api
  labels:
    app: starlight-bitcoin-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: starlight-bitcoin-api
  template:
    metadata:
      labels:
        app: starlight-bitcoin-api
    spec:
      containers:
      - name: api
        image: starlight-bitcoin-api:latest
        ports:
        - containerPort: 8080
        env:
        - name: API_HOST
          value: "0.0.0.0"
        - name: API_PORT
          value: "8080"
        - name: MODEL_PATH
          value: "/app/models/detector_balanced.onnx"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-volume
          mountPath: /app/models
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: starlight-model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: starlight-bitcoin-api-service
spec:
  selector:
    app: starlight-bitcoin-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

Deploy to Kubernetes:

```bash
# Apply deployment
kubectl apply -f k8s-deployment.yaml

# Check status
kubectl get pods -l app=starlight-bitcoin-api
kubectl logs -f deployment/starlight-bitcoin-api
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | 0.0.0.0 | API server bind address |
| `API_PORT` | 8080 | API server port |
| `API_WORKERS` | 4 | Number of worker processes |
| `BITCOIN_NODE_URL` | https://blockstream.info/api | Bitcoin node API URL |
| `MODEL_PATH` | models/detector_balanced.onnx | Path to scanner model |
| `CONFIDENCE_THRESHOLD` | 0.5 | Default confidence threshold |
| `MAX_IMAGE_SIZE` | 10485760 | Maximum image size (10MB) |
| `RATE_LIMIT_PER_MINUTE` | 60 | API rate limit per minute |
| `LOG_LEVEL` | INFO | Logging level |
| `API_KEY_SECRET` | - | Secret for API key generation |

### Bitcoin Node Options

#### Option 1: Public API (Default)
Uses Blockstream API - no setup required.

#### Option 2: Self-Hosted Bitcoin Core

```bash
# Install Bitcoin Core
sudo apt-get install bitcoin-core

# Configure bitcoin.conf
echo "
rpcuser=your_rpc_user
rpcpassword=your_rpc_password
rpcallowip=127.0.0.1
server=1
" > ~/.bitcoin/bitcoin.conf

# Start Bitcoin Core
bitcoind -daemon

# Update API URL
export BITCOIN_NODE_URL=http://127.0.0.1:8332
```

## Security

### API Authentication

1. **API Key Authentication**:
   ```bash
   # Generate secure API key
   openssl rand -hex 32
   
   # Set in environment
   export API_KEY_SECRET=generated_key
   ```

2. **TLS/SSL**:
   ```bash
   # Generate certificates
   openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
   
   # Configure uvicorn with SSL
   uvicorn bitcoin_api:app --host 0.0.0.0 --port 8443 \
       --ssl-keyfile key.pem --ssl-certfile cert.pem
   ```

3. **Firewall Rules**:
   ```bash
   # Allow API traffic
   sudo ufw allow 8080/tcp
   sudo ufw allow 443/tcp
   ```

### Rate Limiting

Configure rate limiting in `nginx` reverse proxy:

```nginx
http {
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/m;
    
    server {
        listen 80;
        server_name api.starlight.ai;
        
        location / {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://localhost:8080;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
    }
}
```

## Monitoring

### Health Checks

```bash
# Basic health
curl http://localhost:8080/health

# Detailed health with JSON
curl http://localhost:8080/health | jq

# Readiness probe
curl http://localhost:8080/health/ready

# Liveness probe
curl http://localhost:8080/health/live
```

### Metrics

The API exposes Prometheus metrics at `/metrics`:

```bash
# Get metrics
curl http://localhost:8080/metrics

# Example metrics output
starlight_bitcoin_scans_total{method="transaction",status="success"} 15420
starlight_bitcoin_scan_duration_seconds{method="image",quantile="0.95"} 0.045
starlight_api_requests_total{endpoint="/scan/transaction",status="200"} 1000
```

### Logging

Configure structured logging:

```python
# In production, use structlog
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
```

## Testing

### API Testing

```bash
# Test health endpoint
curl -X GET "http://localhost:8080/health"

# Test transaction scan
curl -X POST "http://localhost:8080/scan/transaction" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "transaction_id": "f4184fc596403b9d638783cf57adfe4c75c605f6356fbc91338530e9831e9e16",
    "extract_images": true,
    "scan_options": {
      "extract_message": true,
      "confidence_threshold": 0.5
    }
  }'

# Test image upload
curl -X POST "http://localhost:8080/scan/image" \
  -H "Authorization: Bearer your-api-key" \
  -F "image=@test_image.png" \
  -F "extract_message=true"
```

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 -H "Authorization: Bearer your-api-key" \
   http://localhost:8080/health

# Using wrk
wrk -t12 -c400 -d30s --header "Authorization: Bearer your-api-key" \
   http://localhost:8080/health
```

## Performance Tuning

### Scanner Optimization

```bash
# Optimize for GPU (if available)
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="7.5"

# Optimize worker count
export SCANNER_WORKERS=$(nproc)
```

### API Server Tuning

```bash
# Use uvicorn with optimized settings
uvicorn bitcoin_api:app \
    --host 0.0.0.0 \
    --port 8080 \
    --workers $(nproc) \
    --worker-class uvicorn.workers.UvicornWorker \
    --access-log \
    --log-level info \
    --loop uvloop \
    --http httptools
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   ```bash
   # Check model file exists
   ls -la models/detector_balanced.onnx
   
   # Check permissions
   chmod 644 models/detector_balanced.onnx
   ```

2. **Bitcoin Node Connection**:
   ```bash
   # Test Bitcoin API connectivity
   curl -s https://blockstream.info/api/blocks/tip/height
   
   # Check timeout settings
   curl -m 5 https://blockstream.info/api/
   ```

3. **Memory Issues**:
   ```bash
   # Monitor memory usage
   ps aux | grep bitcoin_api
   
   # Adjust worker count
   export SCANNER_WORKERS=2
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with single worker for debugging
uvicorn bitcoin_api:app --host 0.0.0.0 --port 8080 --workers 1 --log-level debug
```

## Production Checklist

- [ ] SSL/TLS certificates configured
- [ ] API keys generated and secured
- [ ] Firewall rules configured
- [ ] Rate limiting enabled
- [ ] Health checks configured
- [ ] Monitoring and alerting set up
- [ ] Log rotation configured
- [ ] Backup strategy for models
- [ ] Load testing completed
- [ ] Documentation updated
- [ ] Security audit completed

## Scaling

### Horizontal Scaling

```bash
# Deploy multiple instances
kubectl scale deployment starlight-bitcoin-api --replicas=5

# Use load balancer
kubectl apply -f load-balancer.yaml
```

### Vertical Scaling

```yaml
# Increase resource limits
resources:
  requests:
    memory: "8Gi"
    cpu: "4000m"
  limits:
    memory: "16Gi"
    cpu: "8000m"
```

## Support

For deployment issues:
- Check logs: `kubectl logs deployment/starlight-bitcoin-api`
- Review health: `curl http://localhost:8080/health`
- Monitor metrics: `curl http://localhost:8080/metrics`
- Documentation: `/docs` endpoint when running