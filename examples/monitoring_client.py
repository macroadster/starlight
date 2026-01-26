"""
Starlight Monitoring Client Example
Client library for interacting with the Starlight Monitoring API

This example demonstrates how to:
1. Log inference metrics
2. Retrieve aggregated statistics
3. Monitor alerts
4. Manage model versions
"""

import requests
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StarlightMonitoringClient:
    """
    Client for interacting with the Starlight Monitoring API
    """

    def __init__(self, base_url: str, api_key: str, timeout: int = 30):
        """
        Initialize the monitoring client

        Args:
            base_url: Base URL of the monitoring API (e.g., "https://api.starlight.ai/monitoring/v1")
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "StarlightMonitoringClient/1.0",
            }
        )

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make an HTTP request to the API

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            **kwargs: Additional arguments for requests

        Returns:
            Response JSON as dictionary

        Raises:
            requests.RequestException: If the request fails
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(
                method=method, url=url, timeout=self.timeout, **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {method} {url} - {e}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the monitoring service

        Returns:
            Health status information
        """
        return self._make_request("GET", "/health")

    def log_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log inference metrics

        Args:
            metrics: Metrics data to log

        Returns:
            Response from the metrics logging endpoint
        """
        # Ensure required fields are present
        required_fields = [
            "timestamp",
            "request_id",
            "model_version",
            "inference",
            "system",
            "metadata",
        ]
        for field in required_fields:
            if field not in metrics:
                raise ValueError(f"Missing required field: {field}")

        return self._make_request("POST", "/metrics/log", json=metrics)

    def get_statistics(
        self,
        window: str = "1h",
        model_version: Optional[str] = None,
        granularity: str = "5m",
    ) -> Dict[str, Any]:
        """
        Retrieve aggregated statistics

        Args:
            window: Time window ('1h', '6h', '24h', '7d')
            model_version: Filter by model version
            granularity: Data granularity ('1m', '5m', '1h')

        Returns:
            Aggregated statistics
        """
        params = {"window": window, "granularity": granularity}
        if model_version:
            params["model_version"] = model_version

        return self._make_request("GET", "/metrics/stats", params=params)

    def get_alerts(
        self, severity: Optional[str] = None, limit: int = 50, active_only: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve recent alerts

        Args:
            severity: Filter by severity ('critical', 'warning', 'info')
            limit: Maximum number of alerts to return
            active_only: Only return active alerts

        Returns:
            Recent alerts
        """
        params = {"limit": limit, "active_only": active_only}
        if severity:
            params["severity"] = severity

        return self._make_request("GET", "/alerts/recent", params=params)

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
        comment: Optional[str] = None,
        suppress_until: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Acknowledge an alert

        Args:
            alert_id: ID of the alert to acknowledge
            acknowledged_by: Who is acknowledging the alert
            comment: Optional acknowledgment comment
            suppress_until: Optional suppression timestamp

        Returns:
            Acknowledgment response
        """
        data = {"alert_id": alert_id, "acknowledged_by": acknowledged_by}
        if comment:
            data["comment"] = comment
        if suppress_until:
            data["suppress_until"] = suppress_until.isoformat()

        return self._make_request("POST", "/alerts/acknowledge", json=data)

    def get_model_versions(self) -> Dict[str, Any]:
        """
        Get available model versions

        Returns:
            Model version information
        """
        return self._make_request("GET", "/models/versions")

    def initiate_rollback(
        self,
        target_version: str,
        reason: str,
        initiated_by: str,
        immediate: bool = True,
    ) -> Dict[str, Any]:
        """
        Initiate a model rollback

        Args:
            target_version: Target version to rollback to
            reason: Reason for rollback
            initiated_by: Who initiated the rollback
            immediate: Whether to perform immediate rollback

        Returns:
            Rollback operation response
        """
        data = {
            "target_version": target_version,
            "reason": reason,
            "initiated_by": initiated_by,
            "immediate": immediate,
        }

        return self._make_request("POST", "/models/rollback", json=data)


class MetricsCollector:
    """
    Utility class for collecting and formatting metrics
    """

    @staticmethod
    def create_inference_metrics(
        request_id: str,
        model_version: str,
        latency_ms: float,
        confidence_score: float,
        prediction: str,
        image_size_bytes: int,
        processing_time_ms: float,
        pod_id: str,
        node_id: str,
        cpu_usage: float,
        memory_usage_mb: float,
        source_ip: str,
        user_agent: str,
    ) -> Dict[str, Any]:
        """
        Create a complete metrics payload

        Args:
            request_id: Unique request identifier
            model_version: Model version used
            latency_ms: Inference latency in milliseconds
            confidence_score: Confidence score (0-1)
            prediction: Model prediction ('stego' or 'clean')
            image_size_bytes: Image size in bytes
            processing_time_ms: Processing time in milliseconds
            pod_id: Kubernetes pod ID
            node_id: Kubernetes node ID
            cpu_usage: CPU usage (0-1)
            memory_usage_mb: Memory usage in MB
            source_ip: Client source IP
            user_agent: Client user agent

        Returns:
            Formatted metrics payload
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "model_version": model_version,
            "inference": {
                "latency_ms": latency_ms,
                "confidence_score": confidence_score,
                "prediction": prediction,
                "image_size_bytes": image_size_bytes,
                "processing_time_ms": processing_time_ms,
            },
            "system": {
                "pod_id": pod_id,
                "node_id": node_id,
                "cpu_usage": cpu_usage,
                "memory_usage_mb": memory_usage_mb,
                "gpu_usage": 0.0,  # Default to 0 if not using GPU
            },
            "metadata": {
                "source_ip": source_ip,
                "user_agent": user_agent,
                "request_type": "api",
            },
        }


class AlertMonitor:
    """
    Utility class for monitoring and managing alerts
    """

    def __init__(self, client: StarlightMonitoringClient):
        self.client = client

    def monitor_alerts(
        self, check_interval: int = 60, callback: Optional[callable] = None
    ):
        """
        Continuously monitor for new alerts

        Args:
            check_interval: Check interval in seconds
            callback: Optional callback function for new alerts
        """
        logger.info(f"Starting alert monitoring with {check_interval}s interval")

        last_alert_count = 0

        try:
            while True:
                try:
                    alerts_response = self.client.get_alerts(active_only=True)
                    current_alert_count = alerts_response["total_count"]

                    if current_alert_count > last_alert_count:
                        new_alerts = alerts_response["alerts"][
                            : current_alert_count - last_alert_count
                        ]
                        logger.warning(f"Found {len(new_alerts)} new alert(s)")

                        for alert in new_alerts:
                            logger.warning(
                                f"ALERT: {alert['title']} - {alert['description']}"
                            )

                            if callback:
                                callback(alert)

                        last_alert_count = current_alert_count

                    time.sleep(check_interval)

                except Exception as e:
                    logger.error(f"Error in alert monitoring loop: {e}")
                    time.sleep(check_interval)

        except KeyboardInterrupt:
            logger.info("Alert monitoring stopped by user")

    def auto_acknowledge_critical(self, acknowledged_by: str, suppress_hours: int = 1):
        """
        Automatically acknowledge critical alerts

        Args:
            acknowledged_by: Who is acknowledging
            suppress_hours: Hours to suppress alerts
        """
        logger.info("Checking for critical alerts to auto-acknowledge")

        try:
            alerts_response = self.client.get_alerts(
                severity="critical", active_only=True
            )

            for alert in alerts_response["alerts"]:
                logger.info(f"Auto-acknowledging critical alert: {alert['id']}")

                suppress_until = datetime.utcnow() + timedelta(hours=suppress_hours)
                self.client.acknowledge_alert(
                    alert_id=alert["id"],
                    acknowledged_by=acknowledged_by,
                    comment="Auto-acknowledged by monitoring system",
                    suppress_until=suppress_until,
                )

        except Exception as e:
            logger.error(f"Error auto-acknowledging alerts: {e}")


def example_usage():
    """
    Example usage of the monitoring client
    """
    # Configuration
    BASE_URL = "https://api.starlight.ai/monitoring/v1"
    API_KEY = "your-api-key-here"

    # Initialize client
    client = StarlightMonitoringClient(BASE_URL, API_KEY)

    try:
        # Health check
        print("=== Health Check ===")
        health = client.health_check()
        print(f"Service status: {health['status']}")
        print(f"Version: {health['version']}")
        print(f"Uptime: {health['uptime_seconds']}s")
        print()

        # Log some sample metrics
        print("=== Logging Metrics ===")
        metrics = MetricsCollector.create_inference_metrics(
            request_id=str(uuid.uuid4()),
            model_version="v4-prod",
            latency_ms=42.5,
            confidence_score=0.87,
            prediction="stego",
            image_size_bytes=1048576,
            processing_time_ms=38.2,
            pod_id="starlight-7d6f8c9c-abcde",
            node_id="worker-node-3",
            cpu_usage=0.35,
            memory_usage_mb=1250,
            source_ip="192.168.1.100",
            user_agent="StarlightClient/1.0",
        )

        response = client.log_metrics(metrics)
        print(f"Metrics logged: {response['status']}")
        print(f"Request ID: {response['request_id']}")
        print()

        # Get statistics
        print("=== Statistics ===")
        stats = client.get_statistics(window="1h")
        print(f"Window: {stats['window']}")
        print(f"Total requests: {stats['statistics']['total_requests']}")
        print(f"Average latency: {stats['statistics']['avg_latency_ms']}ms")
        print(f"Error rate: {stats['statistics']['error_rate']}")
        print(f"Stego rate: {stats['predictions']['stego_rate']}")
        print()

        # Check alerts
        print("=== Alerts ===")
        alerts = client.get_alerts(active_only=True)
        print(f"Active alerts: {alerts['active_count']}")

        for alert in alerts["alerts"][:3]:  # Show first 3 alerts
            print(f"- {alert['severity']}: {alert['title']} ({alert['timestamp']})")
        print()

        # Model versions
        print("=== Model Versions ===")
        models = client.get_model_versions()
        print(f"Current version: {models['current_version']}")

        for version in models["available_versions"]:
            print(
                f"- {version['version']}: {version['status']} (FPR: {version['fpr_rate']})"
            )
        print()

        # Alert monitoring example (commented out to avoid running indefinitely)
        # print("=== Starting Alert Monitor ===")
        # monitor = AlertMonitor(client)
        #
        # def alert_callback(alert):
        #     print(f"New alert: {alert['title']} - {alert['description']}")
        #
        # monitor.monitor_alerts(check_interval=30, callback=alert_callback)

    except Exception as e:
        logger.error(f"Example usage failed: {e}")


def batch_metrics_example():
    """
    Example of logging metrics in batch
    """
    BASE_URL = "https://api.starlight.ai/monitoring/v1"
    API_KEY = "your-api-key-here"

    client = StarlightMonitoringClient(BASE_URL, API_KEY)

    print("=== Batch Metrics Logging ===")

    # Generate sample metrics
    sample_metrics = []
    for i in range(10):
        metrics = MetricsCollector.create_inference_metrics(
            request_id=f"batch-{i}-{uuid.uuid4().hex[:8]}",
            model_version="v4-prod",
            latency_ms=40.0 + (i * 2.5),  # Varying latency
            confidence_score=0.8 + (i * 0.02),
            prediction="stego" if i % 3 == 0 else "clean",
            image_size_bytes=1024 * 1024,  # 1MB
            processing_time_ms=35.0 + (i * 1.5),
            pod_id="starlight-batch-pod",
            node_id="worker-node-1",
            cpu_usage=0.3 + (i * 0.05),
            memory_usage_mb=1200 + (i * 10),
            source_ip="10.0.0.100",
            user_agent="BatchClient/1.0",
        )
        sample_metrics.append(metrics)

    # Log metrics
    successful_logs = 0
    failed_logs = 0

    for metrics in sample_metrics:
        try:
            response = client.log_metrics(metrics)
            successful_logs += 1
            print(f"✓ Logged metrics for request: {metrics['request_id']}")
        except Exception as e:
            failed_logs += 1
            print(f"✗ Failed to log metrics: {e}")

    print(
        f"\nBatch logging complete: {successful_logs} successful, {failed_logs} failed"
    )


def performance_monitoring_example():
    """
    Example of performance monitoring and alerting
    """
    BASE_URL = "https://api.starlight.ai/monitoring/v1"
    API_KEY = "your-api-key-here"

    client = StarlightMonitoringClient(BASE_URL, API_KEY)
    monitor = AlertMonitor(client)

    print("=== Performance Monitoring Example ===")

    # Check current performance
    stats = client.get_statistics(window="1h")
    avg_latency = stats["statistics"]["avg_latency_ms"]
    error_rate = stats["statistics"]["error_rate"]

    print(f"Current average latency: {avg_latency}ms")
    print(f"Current error rate: {error_rate}")

    # Performance thresholds
    LATENCY_THRESHOLD = 100.0  # ms
    ERROR_RATE_THRESHOLD = 0.01  # 1%

    # Check if thresholds are exceeded
    if avg_latency > LATENCY_THRESHOLD:
        print(f"⚠️  High latency detected: {avg_latency}ms > {LATENCY_THRESHOLD}ms")

        # Create a high latency alert (in real scenario, this would be done automatically)
        alert_metrics = MetricsCollector.create_inference_metrics(
            request_id=f"latency-alert-{uuid.uuid4().hex[:8]}",
            model_version="v4-prod",
            latency_ms=avg_latency,
            confidence_score=0.85,
            prediction="clean",
            image_size_bytes=512 * 1024,
            processing_time_ms=avg_latency - 5,
            pod_id="starlight-monitor-pod",
            node_id="worker-node-2",
            cpu_usage=0.45,
            memory_usage_mb=1400,
            source_ip="10.0.0.200",
            user_agent="PerformanceMonitor/1.0",
        )

        try:
            client.log_metrics(alert_metrics)
            print("✓ High latency metrics logged")
        except Exception as e:
            print(f"✗ Failed to log high latency metrics: {e}")

    if error_rate > ERROR_RATE_THRESHOLD:
        print(f"⚠️  High error rate detected: {error_rate} > {ERROR_RATE_THRESHOLD}")

    # Auto-acknowledge critical alerts if needed
    monitor.auto_acknowledge_critical(
        acknowledged_by="performance-monitor", suppress_hours=2
    )


if __name__ == "__main__":
    print("Starlight Monitoring Client Examples")
    print("=" * 50)

    # Run basic example
    print("\n1. Basic Usage Example:")
    example_usage()

    # Run batch metrics example
    print("\n2. Batch Metrics Example:")
    batch_metrics_example()

    # Run performance monitoring example
    print("\n3. Performance Monitoring Example:")
    performance_monitoring_example()

    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo run the alert monitor continuously, uncomment the")
    print("monitor_alerts() call in the example_usage() function.")
