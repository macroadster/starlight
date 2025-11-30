# Starlight Monitoring API Specification

**Version**: 1.0  
**Format**: OpenAPI 3.0  
**Updated**: November 25, 2025  

---

## 🎯 Overview

The Starlight Monitoring API provides real-time metrics collection, alerting, and operational visibility for the steganography detection service. This API is designed for high-throughput, low-latency monitoring with SLA guarantees.

### Base URL
```
Production: https://api.starlight.ai/monitoring/v1
Development: http://localhost:8080/monitoring/v1
```

### Authentication
- **Bearer Token**: JWT-based authentication for external integrations
- **Service-to-Service**: mTLS for internal service communication
- **Public Endpoints**: `/health` and `/metrics/stats` (read-only)

---

## 📊 API Endpoints

### Health & Status

#### GET /health
Service health check with basic metrics.

**SLA**: <100ms response time

**Responses**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-25T10:00:00Z",
  "version": "v4-prod",
  "uptime_seconds": 86400,
  "metrics": {
    "requests_total": 1500000,
    "requests_per_second": 17.5,
    "error_rate": 0.002,
    "avg_latency_ms": 45.2
  },
  "dependencies": {
    "database": "healthy",
    "redis": "healthy",
    "model_store": "healthy"
  }
}
```

**Status Codes**:
- `200 OK` - Service healthy
- `503 Service Unavailable` - Service unhealthy

---

### Metrics Collection

#### POST /metrics/log
Log inference metrics for monitoring and alerting.

**SLA**: <50ms response time

**Request Body**:
```json
{
  "timestamp": "2025-11-25T10:00:00Z",
  "request_id": "req_123456789",
  "model_version": "v4-prod",
  "inference": {
    "latency_ms": 42.5,
    "confidence_score": 0.87,
    "prediction": "stego",
    "image_size_bytes": 1048576,
    "processing_time_ms": 38.2
  },
  "system": {
    "pod_id": "starlight-7d6f8c9c-abcde",
    "node_id": "worker-node-3",
    "cpu_usage": 0.35,
    "memory_usage_mb": 1250,
    "gpu_usage": 0.0
  },
  "metadata": {
    "source_ip": "192.168.1.100",
    "user_agent": "StarlightClient/1.0",
    "request_type": "api"
  }
}
```

**Responses**:
```json
{
  "status": "accepted",
  "request_id": "req_123456789",
  "processed_at": "2025-11-25T10:00:01Z"
}
```

**Status Codes**:
- `202 Accepted` - Metrics logged successfully
- `400 Bad Request` - Invalid metrics format
- `429 Too Many Requests` - Rate limit exceeded

---

#### GET /metrics/stats
Retrieve aggregated statistics for a time window.

**SLA**: <500ms response time

**Query Parameters**:
- `window` (string, required): Time window (`1h`, `6h`, `24h`, `7d`)
- `model_version` (string, optional): Filter by model version
- `granularity` (string, optional): Data granularity (`1m`, `5m`, `1h`)

**Response**:
```json
{
  "window": "1h",
  "granularity": "5m",
  "model_version": "v4-prod",
  "generated_at": "2025-11-25T10:00:00Z",
  "statistics": {
    "total_requests": 63000,
    "successful_requests": 62987,
    "failed_requests": 13,
    "error_rate": 0.0002,
    "avg_latency_ms": 43.8,
    "p95_latency_ms": 78.5,
    "p99_latency_ms": 125.3,
    "throughput_rps": 17.5
  },
  "predictions": {
    "stego_count": 1254,
    "clean_count": 61733,
    "stego_rate": 0.0199,
    "avg_confidence": 0.842
  },
  "system": {
    "avg_cpu_usage": 0.32,
    "avg_memory_usage_mb": 1180,
    "max_memory_usage_mb": 1450,
    "pod_restarts": 0
  },
  "time_series": [
    {
      "timestamp": "2025-11-25T09:00:00Z",
      "requests": 1050,
      "avg_latency_ms": 42.1,
      "error_rate": 0.001
    }
  ]
}
```

---

### Alert Management

#### GET /alerts/recent
Retrieve recent alerts with filtering options.

**SLA**: <200ms response time

**Query Parameters**:
- `severity` (string, optional): Filter by severity (`critical`, `warning`, `info`)
- `limit` (integer, optional): Maximum number of alerts (default: 50)
- `active_only` (boolean, optional): Only active alerts (default: true)

**Response**:
```json
{
  "total_count": 3,
  "active_count": 1,
  "alerts": [
    {
      "id": "alert_789",
      "timestamp": "2025-11-25T09:45:00Z",
      "severity": "warning",
      "status": "active",
      "title": "High Latency Detected",
      "description": "P95 latency exceeded 100ms threshold",
      "source": "prometheus",
      "metric": "starlight_inference_duration_seconds",
      "threshold": 100.0,
      "current_value": 125.3,
      "affected_pods": ["starlight-7d6f8c9c-abcde"],
      "actions": [
        {
          "type": "auto_scale",
          "executed_at": "2025-11-25T09:46:00Z",
          "result": "success"
        }
      ],
      "metadata": {
        "runbook_url": "https://docs.starlight.ai/runbooks/high-latency",
        "escalation_policy": "tier_2"
      }
    }
  ]
}
```

---

#### POST /alerts/acknowledge
Acknowledge an alert to suppress notifications.

**Request Body**:
```json
{
  "alert_id": "alert_789",
  "acknowledged_by": "ops-engineer-1",
  "comment": "Investigating cause, will update in 15 minutes",
  "suppress_until": "2025-11-25T11:00:00Z"
}
```

**Response**:
```json
{
  "status": "acknowledged",
  "alert_id": "alert_789",
  "acknowledged_at": "2025-11-25T10:00:00Z",
  "acknowledged_by": "ops-engineer-1",
  "suppress_until": "2025-11-25T11:00:00Z"
}
```

---

### Model Management

#### GET /models/versions
List available model versions with status.

**SLA**: <100ms response time

**Response**:
```json
{
  "current_version": "v4-prod",
  "available_versions": [
    {
      "version": "v4-prod",
      "status": "active",
      "deployed_at": "2025-11-25T08:00:00Z",
      "model_path": "s3://starlight-models/v4/detector.onnx",
      "file_size_bytes": 52428800,
      "fpr_rate": 0.0007,
      "accuracy": 0.985,
      "supported": true
    },
    {
      "version": "v3-stable",
      "status": "rollback",
      "deployed_at": "2025-11-20T10:00:00Z",
      "model_path": "s3://starlight-models/v3/detector.onnx",
      "file_size_bytes": 45678912,
      "fpr_rate": 0.0039,
      "accuracy": 0.967,
      "supported": true
    }
  ],
  "deployment_history": [
    {
      "version": "v4-prod",
      "deployed_at": "2025-11-25T08:00:00Z",
      "deployed_by": "automated-deployment",
      "previous_version": "v3-stable",
      "rollback_available": true
    }
  ]
}
```

---

#### POST /models/rollback
Initiate rollback to previous model version.

**Request Body**:
```json
{
  "target_version": "v3-stable",
  "reason": "High latency detected in production",
  "initiated_by": "ops-engineer-1",
  "immediate": true
}
```

**Response**:
```json
{
  "status": "initiated",
  "rollback_id": "rollback_456",
  "target_version": "v3-stable",
  "current_version": "v4-prod",
  "estimated_duration_seconds": 300,
  "initiated_at": "2025-11-25T10:00:00Z",
  "steps": [
    {
      "step": 1,
      "description": "Scale down current deployment",
      "status": "pending"
    },
    {
      "step": 2,
      "description": "Deploy target version",
      "status": "pending"
    },
    {
      "step": 3,
      "description": "Verify health checks",
      "status": "pending"
    }
  ]
}
```

---

## 🔒 Rate Limiting

| Endpoint | Rate Limit | Burst | Time Window |
|----------|------------|-------|-------------|
| POST /metrics/log | 1000 req/min | 100 | 1 minute |
| GET /metrics/stats | 100 req/min | 20 | 1 minute |
| GET /alerts/recent | 60 req/min | 10 | 1 minute |
| POST /alerts/acknowledge | 10 req/min | 5 | 1 minute |
| GET /models/versions | 60 req/min | 10 | 1 minute |
| POST /models/rollback | 5 req/min | 2 | 1 minute |

---

## 📝 Data Models

### MetricLog
```json
{
  "timestamp": "string (ISO 8601)",
  "request_id": "string (UUID)",
  "model_version": "string",
  "inference": {
    "latency_ms": "number",
    "confidence_score": "number (0-1)",
    "prediction": "string (stego|clean)",
    "image_size_bytes": "integer",
    "processing_time_ms": "number"
  },
  "system": {
    "pod_id": "string",
    "node_id": "string",
    "cpu_usage": "number (0-1)",
    "memory_usage_mb": "number",
    "gpu_usage": "number (0-1)"
  },
  "metadata": {
    "source_ip": "string",
    "user_agent": "string",
    "request_type": "string"
  }
}
```

### Alert
```json
{
  "id": "string",
  "timestamp": "string (ISO 8601)",
  "severity": "string (critical|warning|info)",
  "status": "string (active|acknowledged|resolved)",
  "title": "string",
  "description": "string",
  "source": "string",
  "metric": "string",
  "threshold": "number",
  "current_value": "number",
  "affected_pods": ["string"],
  "actions": [
    {
      "type": "string",
      "executed_at": "string (ISO 8601)",
      "result": "string"
    }
  ],
  "metadata": {
    "runbook_url": "string",
    "escalation_policy": "string"
  }
}
```

### ModelVersion
```json
{
  "version": "string",
  "status": "string (active|rollback|deprecated)",
  "deployed_at": "string (ISO 8601)",
  "model_path": "string",
  "file_size_bytes": "integer",
  "fpr_rate": "number",
  "accuracy": "number",
  "supported": "boolean"
}
```

---

## 🚨 Error Responses

### Standard Error Format
```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": "object",
    "timestamp": "string (ISO 8601)",
    "request_id": "string"
  }
}
```

### Common Error Codes
| Code | HTTP Status | Description |
|------|------------|-------------|
| `INVALID_REQUEST` | 400 | Request body validation failed |
| `MISSING_PARAMETER` | 400 | Required parameter missing |
| `UNAUTHORIZED` | 401 | Authentication failed |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Server internal error |
| `SERVICE_UNAVAILABLE` | 503 | Dependent service unavailable |

---

## 🔗 Integration Examples

### Python Client
```python
import requests
import json
from datetime import datetime

class StarlightMonitoringClient:
    def __init__(self, base_url, api_key=None):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'StarlightMonitoringClient/1.0'
        }
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
    
    def log_metrics(self, metrics):
        """Log inference metrics"""
        response = requests.post(
            f'{self.base_url}/metrics/log',
            headers=self.headers,
            json=metrics
        )
        response.raise_for_status()
        return response.json()
    
    def get_stats(self, window='1h', model_version=None):
        """Get aggregated statistics"""
        params = {'window': window}
        if model_version:
            params['model_version'] = model_version
        
        response = requests.get(
            f'{self.base_url}/metrics/stats',
            headers=self.headers,
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def get_alerts(self, severity=None, limit=50, active_only=True):
        """Get recent alerts"""
        params = {'limit': limit, 'active_only': active_only}
        if severity:
            params['severity'] = severity
        
        response = requests.get(
            f'{self.base_url}/alerts/recent',
            headers=self.headers,
            params=params
        )
        response.raise_for_status()
        return response.json()

# Usage example
client = StarlightMonitoringClient('https://api.starlight.ai/monitoring/v1')

# Log metrics
metrics = {
    "timestamp": datetime.utcnow().isoformat(),
    "request_id": "req_123456789",
    "model_version": "v4-prod",
    "inference": {
        "latency_ms": 42.5,
        "confidence_score": 0.87,
        "prediction": "stego",
        "image_size_bytes": 1048576,
        "processing_time_ms": 38.2
    },
    "system": {
        "pod_id": "starlight-7d6f8c9c-abcde",
        "node_id": "worker-node-3",
        "cpu_usage": 0.35,
        "memory_usage_mb": 1250,
        "gpu_usage": 0.0
    }
}
client.log_metrics(metrics)

# Get statistics
stats = client.get_stats(window='1h')
print(f"Total requests: {stats['statistics']['total_requests']}")
print(f"Average latency: {stats['statistics']['avg_latency_ms']}ms")
```

### cURL Examples
```bash
# Health check
curl -X GET "https://api.starlight.ai/monitoring/v1/health"

# Log metrics
curl -X POST "https://api.starlight.ai/monitoring/v1/metrics/log" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "timestamp": "2025-11-25T10:00:00Z",
    "request_id": "req_123456789",
    "model_version": "v4-prod",
    "inference": {
      "latency_ms": 42.5,
      "confidence_score": 0.87,
      "prediction": "stego"
    }
  }'

# Get statistics
curl -X GET "https://api.starlight.ai/monitoring/v1/metrics/stats?window=1h" \
  -H "Authorization: Bearer YOUR_API_KEY"

# Get alerts
curl -X GET "https://api.starlight.ai/monitoring/v1/alerts/recent?severity=warning" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

## 📊 Monitoring & Observability

### API Metrics
The Monitoring API itself exposes metrics at `/metrics` (Prometheus format):

```
# API request metrics
starlight_monitoring_requests_total{endpoint="/metrics/log", status="202"} 15420
starlight_monitoring_request_duration_seconds{endpoint="/metrics/stats", quantile="0.95"} 0.045

# Business metrics
starlight_inference_requests_total{model_version="v4-prod"} 1500000
starlight_inference_latency_seconds{model_version="v4-prod", quantile="0.95"} 0.078
starlight_false_positive_rate{model_version="v4-prod"} 0.0007
```

### Health Check Endpoints
- `/health` - Basic health status
- `/health/ready` - Readiness probe (dependencies healthy)
- `/health/live` - Liveness probe (service responsive)
- `/metrics` - Prometheus metrics endpoint

---

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-25 | Initial release with core monitoring endpoints |
| 0.9 | 2025-11-20 | Beta testing with internal teams |
| 0.8 | 2025-11-15 | Added alert management endpoints |
| 0.7 | 2025-11-10 | Added model rollback functionality |

---

## 📞 Support & Contact

### API Support
- **Documentation**: https://docs.starlight.ai/monitoring-api
- **Issues**: api-issues@starlight.ai
- **Status Page**: https://status.starlight.ai

### SLA Information
- **Availability**: 99.9% (monthly)
- **Support Response**: <1 hour for P1 incidents
- **Maintenance Window**: Sundays 02:00-04:00 UTC

---

**Specification Version**: 1.0  
**Last Updated**: November 25, 2025  
**Next Review**: December 25, 2025  
**Maintainer**: GPT-OSS (Documentation & API Infrastructure)