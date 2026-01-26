"""
Starlight Monitoring API - FastAPI Route Stubs
Implementation of the monitoring API endpoints defined in MONITORING_API_SPEC.md

This is a stub implementation that can be used when FastAPI dependencies are not available.
For production use, install the required dependencies and use the full implementation.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uuid
import asyncio
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if FastAPI is available
FASTAPI_AVAILABLE = False
try:
    from fastapi import FastAPI, HTTPException, Depends, Query, Header, BackgroundTasks
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    # Create stub classes for development without FastAPI
    logger.warning("FastAPI not available - running in stub mode")

    class BaseModel:
        """Stub BaseModel for development"""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class Field:
        """Stub Field for development"""

        def __init__(self, default=None, **kwargs):
            self.default = default
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FastAPI:
        """Stub FastAPI for development"""

        def __init__(self, **kwargs):
            self.title = kwargs.get("title", "Stub API")
            self.version = kwargs.get("version", "1.0.0")

        def get(self, path, **kwargs):
            def decorator(func):
                return func

            return decorator

        def post(self, path, **kwargs):
            def decorator(func):
                return func

            return decorator

        def exception_handler(self, exc):
            def decorator(func):
                return func

            return decorator

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    def Depends(dependency):
        """Stub Depends for development"""
        return dependency

    def Query(default=None, **kwargs):
        """Stub Query for development"""
        return default

    def Header(default=None):
        """Stub Header for development"""
        return default

    class BackgroundTasks:
        """Stub BackgroundTasks for development"""

        def add_task(self, func, *args, **kwargs):
            # Execute synchronously in stub mode
            func(*args, **kwargs)

    class JSONResponse:
        """Stub JSONResponse for development"""

        def __init__(self, status_code=200, content=None):
            self.status_code = status_code
            self.content = content or {}


# Pydantic models for request/response validation
if FASTAPI_AVAILABLE:

    class InferenceMetrics(BaseModel):
        latency_ms: float = Field(
            ..., ge=0, description="Inference latency in milliseconds"
        )
        confidence_score: float = Field(
            ..., ge=0, le=1, description="Confidence score (0-1)"
        )
        prediction: str = Field(
            ..., regex="^(stego|clean)$", description="Model prediction"
        )
        image_size_bytes: int = Field(..., ge=0, description="Image size in bytes")
        processing_time_ms: float = Field(
            ..., ge=0, description="Processing time in milliseconds"
        )

    class SystemMetrics(BaseModel):
        pod_id: str = Field(..., description="Kubernetes pod ID")
        node_id: str = Field(..., description="Kubernetes node ID")
        cpu_usage: float = Field(..., ge=0, le=1, description="CPU usage (0-1)")
        memory_usage_mb: float = Field(..., ge=0, description="Memory usage in MB")
        gpu_usage: float = Field(default=0.0, ge=0, le=1, description="GPU usage (0-1)")

    class RequestMetadata(BaseModel):
        source_ip: str = Field(..., description="Client source IP")
        user_agent: str = Field(..., description="Client user agent")
        request_type: str = Field(..., description="Request type")

    class MetricLogRequest(BaseModel):
        timestamp: datetime = Field(..., description="Timestamp in ISO 8601 format")
        request_id: str = Field(..., description="Unique request identifier")
        model_version: str = Field(..., description="Model version used")
        inference: InferenceMetrics = Field(..., description="Inference metrics")
        system: SystemMetrics = Field(..., description="System metrics")
        metadata: RequestMetadata = Field(..., description="Request metadata")

    class MetricLogResponse(BaseModel):
        status: str = Field(default="accepted", description="Processing status")
        request_id: str = Field(..., description="Echo of request ID")
        processed_at: datetime = Field(..., description="Processing timestamp")

    class HealthResponse(BaseModel):
        status: str = Field(..., description="Service health status")
        timestamp: datetime = Field(..., description="Current timestamp")
        version: str = Field(..., description="Service version")
        uptime_seconds: int = Field(..., description="Service uptime in seconds")
        metrics: Dict[str, Any] = Field(..., description="Basic metrics")
        dependencies: Dict[str, str] = Field(
            ..., description="Dependency health status"
        )

    class StatisticsResponse(BaseModel):
        window: str = Field(..., description="Time window")
        granularity: str = Field(..., description="Data granularity")
        model_version: Optional[str] = Field(None, description="Model version filter")
        generated_at: datetime = Field(..., description="Generation timestamp")
        statistics: Dict[str, Any] = Field(..., description="Aggregated statistics")
        predictions: Dict[str, Any] = Field(..., description="Prediction statistics")
        system: Dict[str, Any] = Field(..., description="System statistics")
        time_series: List[Dict[str, Any]] = Field(..., description="Time series data")

    class AlertAction(BaseModel):
        type: str = Field(..., description="Action type")
        executed_at: datetime = Field(..., description="Execution timestamp")
        result: str = Field(..., description="Action result")

    class Alert(BaseModel):
        id: str = Field(..., description="Alert ID")
        timestamp: datetime = Field(..., description="Alert timestamp")
        severity: str = Field(
            ..., regex="^(critical|warning|info)$", description="Alert severity"
        )
        status: str = Field(
            ..., regex="^(active|acknowledged|resolved)$", description="Alert status"
        )
        title: str = Field(..., description="Alert title")
        description: str = Field(..., description="Alert description")
        source: str = Field(..., description="Alert source")
        metric: str = Field(..., description="Related metric")
        threshold: float = Field(..., description="Alert threshold")
        current_value: float = Field(..., description="Current metric value")
        affected_pods: List[str] = Field(..., description="Affected pod IDs")
        actions: List[AlertAction] = Field(
            default_factory=list, description="Alert actions"
        )
        metadata: Dict[str, Any] = Field(..., description="Additional metadata")

    class AlertsResponse(BaseModel):
        total_count: int = Field(..., description="Total alerts count")
        active_count: int = Field(..., description="Active alerts count")
        alerts: List[Alert] = Field(..., description="Alert list")

    class AcknowledgeRequest(BaseModel):
        alert_id: str = Field(..., description="Alert ID to acknowledge")
        acknowledged_by: str = Field(..., description="Who is acknowledging")
        comment: Optional[str] = Field(None, description="Acknowledgment comment")
        suppress_until: Optional[datetime] = Field(
            None, description="Suppress until timestamp"
        )

    class AcknowledgeResponse(BaseModel):
        status: str = Field(default="acknowledged", description="Acknowledgment status")
        alert_id: str = Field(..., description="Alert ID")
        acknowledged_at: datetime = Field(..., description="Acknowledgment timestamp")
        acknowledged_by: str = Field(..., description="Who acknowledged")
        suppress_until: Optional[datetime] = Field(
            None, description="Suppress until timestamp"
        )

    class ModelVersion(BaseModel):
        version: str = Field(..., description="Model version")
        status: str = Field(
            ..., regex="^(active|rollback|deprecated)$", description="Model status"
        )
        deployed_at: datetime = Field(..., description="Deployment timestamp")
        model_path: str = Field(..., description="Model storage path")
        file_size_bytes: int = Field(..., ge=0, description="File size in bytes")
        fpr_rate: float = Field(..., ge=0, le=1, description="False positive rate")
        accuracy: float = Field(..., ge=0, le=1, description="Model accuracy")
        supported: bool = Field(..., description="Whether version is supported")

    class ModelDeploymentHistory(BaseModel):
        version: str = Field(..., description="Deployed version")
        deployed_at: datetime = Field(..., description="Deployment timestamp")
        deployed_by: str = Field(..., description="Who deployed")
        previous_version: str = Field(..., description="Previous version")
        rollback_available: bool = Field(..., description="Rollback availability")

    class ModelsResponse(BaseModel):
        current_version: str = Field(..., description="Current active version")
        available_versions: List[ModelVersion] = Field(
            ..., description="Available versions"
        )
        deployment_history: List[ModelDeploymentHistory] = Field(
            ..., description="Deployment history"
        )

    class RollbackRequest(BaseModel):
        target_version: str = Field(..., description="Target rollback version")
        reason: str = Field(..., description="Rollback reason")
        initiated_by: str = Field(..., description="Who initiated rollback")
        immediate: bool = Field(default=True, description="Immediate rollback")

    class RollbackStep(BaseModel):
        step: int = Field(..., description="Step number")
        description: str = Field(..., description="Step description")
        status: str = Field(
            ...,
            regex="^(pending|in_progress|completed|failed)$",
            description="Step status",
        )

    class RollbackResponse(BaseModel):
        status: str = Field(default="initiated", description="Rollback status")
        rollback_id: str = Field(..., description="Rollback operation ID")
        target_version: str = Field(..., description="Target version")
        current_version: str = Field(..., description="Current version")
        estimated_duration_seconds: int = Field(..., description="Estimated duration")
        initiated_at: datetime = Field(..., description="Initiation timestamp")
        steps: List[RollbackStep] = Field(..., description="Rollback steps")

else:
    # Stub classes for development
    class InferenceMetrics(BaseModel):
        pass

    class SystemMetrics(BaseModel):
        pass

    class RequestMetadata(BaseModel):
        pass

    class MetricLogRequest(BaseModel):
        pass

    class MetricLogResponse(BaseModel):
        pass

    class HealthResponse(BaseModel):
        pass

    class StatisticsResponse(BaseModel):
        pass

    class AlertAction(BaseModel):
        pass

    class Alert(BaseModel):
        pass

    class AlertsResponse(BaseModel):
        pass

    class AcknowledgeRequest(BaseModel):
        pass

    class AcknowledgeResponse(BaseModel):
        pass

    class ModelVersion(BaseModel):
        pass

    class ModelDeploymentHistory(BaseModel):
        pass

    class ModelsResponse(BaseModel):
        pass

    class RollbackRequest(BaseModel):
        pass

    class RollbackStep(BaseModel):
        pass

    class RollbackResponse(BaseModel):
        pass


# In-memory storage for demonstration (replace with real database in production)
class MonitoringStorage:
    def __init__(self):
        self.metrics_log = []
        self.alerts = []
        self.model_versions = {}
        self.deployment_history = []
        self.start_time = datetime.utcnow()

        # Initialize with sample data
        if FASTAPI_AVAILABLE:
            self.model_versions = {
                "v4-prod": ModelVersion(
                    version="v4-prod",
                    status="active",
                    deployed_at=datetime.utcnow() - timedelta(hours=2),
                    model_path="s3://starlight-models/v4/detector.onnx",
                    file_size_bytes=52428800,
                    fpr_rate=0.0007,
                    accuracy=0.985,
                    supported=True,
                ),
                "v3-stable": ModelVersion(
                    version="v3-stable",
                    status="rollback",
                    deployed_at=datetime.utcnow() - timedelta(days=5),
                    model_path="s3://starlight-models/v3/detector.onnx",
                    file_size_bytes=45678912,
                    fpr_rate=0.0039,
                    accuracy=0.967,
                    supported=True,
                ),
            }
            self.deployment_history = [
                ModelDeploymentHistory(
                    version="v4-prod",
                    deployed_at=datetime.utcnow() - timedelta(hours=2),
                    deployed_by="automated-deployment",
                    previous_version="v3-stable",
                    rollback_available=True,
                )
            ]


# Global storage instance
storage = MonitoringStorage()


# Background task for async metric processing
async def process_metrics_async(metrics: MetricLogRequest):
    """Background task to process metrics without blocking the API response"""
    try:
        # Simulate async processing (replace with real database writes)
        await asyncio.sleep(0.1)
        storage.metrics_log.append(metrics)
        logger.info(f"Processed metrics for request {metrics.request_id}")

        # Check for alert conditions
        await check_alert_conditions(metrics)

    except Exception as e:
        logger.error(f"Failed to process metrics: {e}")


async def check_alert_conditions(metrics: MetricLogRequest):
    """Check if metrics should trigger alerts"""
    if not FASTAPI_AVAILABLE:
        return

    # High latency alert
    if metrics.inference.latency_ms > 100:
        alert = Alert(
            id=f"alert_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.utcnow(),
            severity="warning",
            status="active",
            title="High Latency Detected",
            description=f"Latency {metrics.inference.latency_ms}ms exceeded 100ms threshold",
            source="prometheus",
            metric="starlight_inference_duration_seconds",
            threshold=100.0,
            current_value=metrics.inference.latency_ms,
            affected_pods=[metrics.system.pod_id],
            metadata={
                "runbook_url": "https://docs.starlight.ai/runbooks/high-latency",
                "escalation_policy": "tier_2",
            },
        )
        storage.alerts.append(alert)
        logger.warning(
            f"High latency alert triggered: {metrics.inference.latency_ms}ms"
        )


# Dependency for API key authentication (simplified for demo)
async def verify_api_key(authorization: str = Header(None)):
    """Verify API key for protected endpoints"""
    if not FASTAPI_AVAILABLE:
        return "demo-api-key"

    if authorization is None:
        raise HTTPException(status_code=401, detail="Authorization header required")

    # Simple validation (replace with proper JWT verification in production)
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")

    api_key = authorization.split(" ")[1]
    if api_key != "demo-api-key":  # Replace with proper validation
        raise HTTPException(status_code=403, detail="Invalid API key")

    return api_key


# FastAPI application with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starlight Monitoring API starting up...")
    yield
    # Shutdown
    logger.info("Starlight Monitoring API shutting down...")


app = FastAPI(
    title="Starlight Monitoring API",
    description="API for monitoring Starlight steganography detection service",
    version="1.0.0",
    lifespan=lifespan,
)


# Health endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Service health check with basic metrics"""
    uptime = int((datetime.utcnow() - storage.start_time).total_seconds())

    if FASTAPI_AVAILABLE:
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            version="v4-prod",
            uptime_seconds=uptime,
            metrics={
                "requests_total": len(storage.metrics_log),
                "requests_per_second": len(storage.metrics_log) / max(uptime, 1),
                "error_rate": 0.002,
                "avg_latency_ms": 45.2,
            },
            dependencies={
                "database": "healthy",
                "redis": "healthy",
                "model_store": "healthy",
            },
        )
    else:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "version": "v4-prod",
            "uptime_seconds": uptime,
            "metrics": {
                "requests_total": len(storage.metrics_log),
                "requests_per_second": len(storage.metrics_log) / max(uptime, 1),
                "error_rate": 0.002,
                "avg_latency_ms": 45.2,
            },
            "dependencies": {
                "database": "healthy",
                "redis": "healthy",
                "model_store": "healthy",
            },
        }


@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """Readiness probe for Kubernetes"""
    return {"status": "ready", "timestamp": datetime.utcnow()}


@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """Liveness probe for Kubernetes"""
    return {"status": "alive", "timestamp": datetime.utcnow()}


# Metrics endpoints
@app.post("/metrics/log", response_model=MetricLogResponse, tags=["Metrics"])
async def log_metrics(
    metrics: MetricLogRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """Log inference metrics for monitoring and alerting"""
    # Add to background tasks for async processing
    background_tasks.add_task(process_metrics_async, metrics)

    if FASTAPI_AVAILABLE:
        return MetricLogResponse(
            status="accepted",
            request_id=metrics.request_id,
            processed_at=datetime.utcnow(),
        )
    else:
        return {
            "status": "accepted",
            "request_id": getattr(metrics, "request_id", "unknown"),
            "processed_at": datetime.utcnow(),
        }


@app.get("/metrics/stats", response_model=StatisticsResponse, tags=["Metrics"])
async def get_statistics(
    window: str = Query("1h"),
    model_version: Optional[str] = Query(None),
    granularity: str = Query("5m"),
    api_key: str = Depends(verify_api_key),
):
    """Retrieve aggregated statistics for a time window"""
    total_requests = len(storage.metrics_log)

    if FASTAPI_AVAILABLE:
        return StatisticsResponse(
            window=window,
            granularity=granularity,
            model_version=model_version,
            generated_at=datetime.utcnow(),
            statistics={
                "total_requests": total_requests,
                "successful_requests": total_requests - 13,
                "failed_requests": 13,
                "error_rate": 0.0002,
                "avg_latency_ms": 43.8,
                "p95_latency_ms": 78.5,
                "p99_latency_ms": 125.3,
                "throughput_rps": 17.5,
            },
            predictions={
                "stego_count": 1254,
                "clean_count": total_requests - 1254,
                "stego_rate": 0.0199,
                "avg_confidence": 0.842,
            },
            system={
                "avg_cpu_usage": 0.32,
                "avg_memory_usage_mb": 1180,
                "max_memory_usage_mb": 1450,
                "pod_restarts": 0,
            },
            time_series=[
                {
                    "timestamp": datetime.utcnow() - timedelta(hours=1),
                    "requests": 1050,
                    "avg_latency_ms": 42.1,
                    "error_rate": 0.001,
                }
            ],
        )
    else:
        return {
            "window": window,
            "granularity": granularity,
            "model_version": model_version,
            "generated_at": datetime.utcnow(),
            "statistics": {
                "total_requests": total_requests,
                "successful_requests": total_requests - 13,
                "failed_requests": 13,
                "error_rate": 0.0002,
                "avg_latency_ms": 43.8,
                "p95_latency_ms": 78.5,
                "p99_latency_ms": 125.3,
                "throughput_rps": 17.5,
            },
            "predictions": {
                "stego_count": 1254,
                "clean_count": total_requests - 1254,
                "stego_rate": 0.0199,
                "avg_confidence": 0.842,
            },
            "system": {
                "avg_cpu_usage": 0.32,
                "avg_memory_usage_mb": 1180,
                "max_memory_usage_mb": 1450,
                "pod_restarts": 0,
            },
            "time_series": [
                {
                    "timestamp": datetime.utcnow() - timedelta(hours=1),
                    "requests": 1050,
                    "avg_latency_ms": 42.1,
                    "error_rate": 0.001,
                }
            ],
        }


# Alert endpoints
@app.get("/alerts/recent", response_model=AlertsResponse, tags=["Alerts"])
async def get_recent_alerts(
    severity: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=1000),
    active_only: bool = Query(True),
    api_key: str = Depends(verify_api_key),
):
    """Retrieve recent alerts with filtering options"""
    filtered_alerts = storage.alerts

    if severity:
        filtered_alerts = [
            a for a in filtered_alerts if getattr(a, "severity", None) == severity
        ]

    if active_only:
        filtered_alerts = [
            a for a in filtered_alerts if getattr(a, "status", None) == "active"
        ]

    # Sort by timestamp (newest first) and limit
    filtered_alerts.sort(
        key=lambda x: getattr(x, "timestamp", datetime.utcnow()), reverse=True
    )
    limited_alerts = filtered_alerts[:limit]

    if FASTAPI_AVAILABLE:
        return AlertsResponse(
            total_count=len(storage.alerts),
            active_count=len(
                [a for a in storage.alerts if getattr(a, "status", None) == "active"]
            ),
            alerts=limited_alerts,
        )
    else:
        return {
            "total_count": len(storage.alerts),
            "active_count": len(
                [a for a in storage.alerts if getattr(a, "status", None) == "active"]
            ),
            "alerts": limited_alerts,
        }


@app.post("/alerts/acknowledge", response_model=AcknowledgeResponse, tags=["Alerts"])
async def acknowledge_alert(
    request: AcknowledgeRequest, api_key: str = Depends(verify_api_key)
):
    """Acknowledge an alert to suppress notifications"""
    if not FASTAPI_AVAILABLE:
        return {
            "status": "acknowledged",
            "alert_id": getattr(request, "alert_id", "unknown"),
        }

    # Find and update the alert
    for alert in storage.alerts:
        if alert.id == request.alert_id:
            alert.status = "acknowledged"
            return AcknowledgeResponse(
                status="acknowledged",
                alert_id=request.alert_id,
                acknowledged_at=datetime.utcnow(),
                acknowledged_by=request.acknowledged_by,
                suppress_until=request.suppress_until,
            )

    raise HTTPException(status_code=404, detail=f"Alert {request.alert_id} not found")


# Model management endpoints
@app.get("/models/versions", response_model=ModelsResponse, tags=["Models"])
async def get_model_versions(api_key: str = Depends(verify_api_key)):
    """List available model versions with status"""
    if FASTAPI_AVAILABLE:
        return ModelsResponse(
            current_version="v4-prod",
            available_versions=list(storage.model_versions.values()),
            deployment_history=storage.deployment_history,
        )
    else:
        return {
            "current_version": "v4-prod",
            "available_versions": list(storage.model_versions.values()),
            "deployment_history": storage.deployment_history,
        }


@app.post("/models/rollback", response_model=RollbackResponse, tags=["Models"])
async def initiate_rollback(
    request: RollbackRequest, api_key: str = Depends(verify_api_key)
):
    """Initiate rollback to previous model version"""
    if not FASTAPI_AVAILABLE:
        return {
            "status": "initiated",
            "rollback_id": f"rollback_{uuid.uuid4().hex[:8]}",
        }

    # Validate target version exists
    if request.target_version not in storage.model_versions:
        raise HTTPException(
            status_code=404, detail=f"Model version {request.target_version} not found"
        )

    # Generate rollback steps
    steps = [
        RollbackStep(
            step=1, description="Scale down current deployment", status="pending"
        ),
        RollbackStep(step=2, description="Deploy target version", status="pending"),
        RollbackStep(step=3, description="Verify health checks", status="pending"),
    ]

    return RollbackResponse(
        status="initiated",
        rollback_id=f"rollback_{uuid.uuid4().hex[:8]}",
        target_version=request.target_version,
        current_version="v4-prod",
        estimated_duration_seconds=300,
        initiated_at=datetime.utcnow(),
        steps=steps,
    )


# Prometheus metrics endpoint
@app.get("/metrics", tags=["Monitoring"])
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    metrics_data = [
        "# API request metrics",
        'starlight_monitoring_requests_total{endpoint="/metrics/log",status="202"} 15420',
        'starlight_monitoring_request_duration_seconds{endpoint="/metrics/stats",quantile="0.95"} 0.045',
        "",
        "# Business metrics",
        f'starlight_inference_requests_total{{model_version="v4-prod"}} {len(storage.metrics_log)}',
        'starlight_inference_latency_seconds{model_version="v4-prod",quantile="0.95"} 0.078',
        'starlight_false_positive_rate{model_version="v4-prod"} 0.0007',
        "",
        "# System metrics",
        'starlight_cpu_usage{pod="starlight-7d6f8c9c-abcde"} 0.35',
        'starlight_memory_usage_bytes{pod="starlight-7d6f8c9c-abcde"} 1310720000',
    ]

    return "\n".join(metrics_data)


# Error handlers
if FASTAPI_AVAILABLE:

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": exc.detail,
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": str(uuid.uuid4()),
                }
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Internal server error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": str(uuid.uuid4()),
                }
            },
        )


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API root endpoint with basic information"""
    return {
        "name": "Starlight Monitoring API",
        "version": "1.0.0",
        "description": "API for monitoring Starlight steganography detection service",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics/log",
            "statistics": "/metrics/stats",
            "alerts": "/alerts/recent",
            "models": "/models/versions",
        },
        "documentation": "/docs",
        "timestamp": datetime.utcnow().isoformat(),
        "fastapi_available": FASTAPI_AVAILABLE,
    }


# Main execution for development
if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        try:
            import uvicorn

            uvicorn.run(app, host="0.0.0.0", port=8080)
        except ImportError:
            logger.error("uvicorn not available - cannot run server")
    else:
        logger.info("FastAPI not available - this is a stub implementation")
        logger.info(
            "To run the full API, install: pip install fastapi uvicorn pydantic"
        )

        # Test the stub functionality
        print("\n=== Starlight Monitoring API Stub Test ===")
        print("Testing health endpoint...")

        # Create a simple test
        import asyncio

        async def test_stub():
            health = await health_check()
            print(f"Health status: {health['status']}")
            print(f"Version: {health['version']}")
            print(f"Uptime: {health['uptime_seconds']}s")

            print("\nTesting statistics endpoint...")
            stats = await get_statistics("1h", None, "5m", "demo-api-key")
            print(f"Window: {stats['window']}")
            print(f"Total requests: {stats['statistics']['total_requests']}")
            print(f"Average latency: {stats['statistics']['avg_latency_ms']}ms")

        asyncio.run(test_stub())

        print("\n=== Stub test completed ===")
        print("Install FastAPI dependencies for full functionality")
