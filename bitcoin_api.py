"""
Starlight Bitcoin Scanning API - FastAPI Implementation
REST API for scanning Bitcoin transactions for steganography in embedded images.

This API integrates with the existing Starlight scanner to provide:
- Transaction scanning for steganography
- Direct image scanning
- Batch processing capabilities
- Message extraction from steganographic images
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import uuid
import logging
import base64
import io
import os
import tempfile
import hashlib
import json
import hmac
import time
from pathlib import Path
import requests
from contextlib import asynccontextmanager
import glob
import threading

from PIL import Image
from starlight.agents.config import Config as AgentConfig
from starlight.agents.client import StargateClient
from starlight.agents.watcher import WatcherAgent
from starlight.agents.worker import WorkerAgent
from starlight.agents.dynamic_loader import dynamic_loader, LoadRequest as DynamicLoadRequest
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_safe_path(path: str, allowed_bases: List[str]) -> str:
    """Validate path is within allowed base directories to prevent directory traversal."""
    abs_path = os.path.abspath(path)
    for base in allowed_bases:
        abs_base = os.path.abspath(base)
        if abs_path == abs_base or abs_path.startswith(abs_base + os.sep):
            return abs_path
    
    # If no match found, use first allowed base as safe default
    return os.path.abspath(allowed_bases[0])

# Default blocks directory (override via env for testing only)
# SECURITY: Restrict to safe subdirectories to prevent path traversal
# Include common Docker mount points and production paths
ALLOWED_BLOCKS_BASES = [
    "blocks", "./blocks", "../blocks", "../../blocks",  # Development paths
    "/data/blocks", "/app/blocks", "/starlight/blocks",  # Docker/container paths
    "/var/lib/starlight/blocks"  # System installation paths
]
_env_blocks = os.environ.get("BLOCKS_DIR", "blocks")
BLOCKS_DIR = validate_safe_path(_env_blocks, ALLOWED_BLOCKS_BASES)
STARGATE_STEGO_CALLBACK_URL = os.environ.get("STARGATE_STEGO_CALLBACK_URL", "")
STARGATE_STEGO_CALLBACK_SECRET = os.environ.get("STARGATE_STEGO_CALLBACK_SECRET", "")


def send_stargate_callback(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send scan results to Stargate callback API when configured."""
    if not STARGATE_STEGO_CALLBACK_URL:
        return {"skipped": True}

    try:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if STARGATE_STEGO_CALLBACK_SECRET:
            signature = hmac.new(
                STARGATE_STEGO_CALLBACK_SECRET.encode("utf-8"),
                body,
                hashlib.sha256,
            ).hexdigest()
            headers["X-Starlight-Signature"] = signature

        resp = requests.post(
            STARGATE_STEGO_CALLBACK_URL, data=body, headers=headers, timeout=10
        )
        return {"status_code": resp.status_code, "response": resp.text}
    except Exception as exc:  # pragma: no cover - network issues are runtime concerns
        logger.warning("Stargate stego callback failed: %s", exc)
        return {"error": str(exc)}


try:
    from fastapi import (
        FastAPI,
        HTTPException,
        Depends,
        Query,
        Header,
        BackgroundTasks,
        File,
        UploadFile,
        Form,
    )
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - FastAPI is required for this service
    raise RuntimeError(
        "FastAPI dependencies are required. Install with `pip install -r bitcoin_api_requirements.txt`."
    ) from exc

# Import scanner components
try:
    from scanner import StarlightScanner, _scan_logic

    SCANNER_AVAILABLE = True
    logger.info("Starlight scanner loaded successfully")
except ImportError as e:
    SCANNER_AVAILABLE = False
    logger.error(f"Could not import Starlight scanner: {e}")

# Stego embedding helpers (re-use existing tool functions when available)
embed_alpha = embed_lsb = embed_palette = embed_exif = embed_eoi = None
try:
    from scripts.stego_tool import (
        embed_alpha,
        embed_lsb,
        embed_palette,
        embed_exif,
        embed_eoi,
    )

    STEGO_HELPERS_AVAILABLE = True
except Exception as e:
    STEGO_HELPERS_AVAILABLE = False
    logger.warning(f"stego_tool helpers unavailable: {e}")


# Bitcoin integration (stub for now - would integrate with actual Bitcoin node)
class BitcoinNodeClient:
    """Bitcoin node client for transaction data retrieval"""

    def __init__(self, node_url: str = "https://blockstream.info/api"):
        self.node_url = node_url
        self.connected = True

    async def get_transaction(self, tx_id: str) -> Dict[str, Any]:
        """Get transaction details from Bitcoin node"""
        # Stub implementation - would integrate with actual Bitcoin API
        return {
            "txid": tx_id,
            "block_height": 170000,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "confirmed",
            "outputs": [
                {
                    "script_pubkey": "OP_RETURN "
                    + "48656c6c6f20576f726c64",  # "Hello World" in hex
                    "value": 0,
                }
            ],
        }

    async def extract_images(self, tx_id: str) -> List[Dict[str, Any]]:
        """Extract images from transaction outputs"""
        # Stub implementation - would parse actual transaction data
        return [
            {
                "index": 0,
                "data": base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                ),
                "format": "png",
                "size_bytes": 67,
            }
        ]

    async def get_block_height(self) -> int:
        """Get current block height"""
        return 856789


# Pydantic models for request/response validation
class ScanOptions(BaseModel):
    extract_message: bool = Field(
        default=True, description="Extract hidden messages if stego detected"
    )
    confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )
    include_metadata: bool = Field(
        default=True, description="Include detailed metadata in response"
    )
    enable_patch_scanning: bool = Field(
        default=True, description="Enable patch-based scanning for large images"
    )
    patch_size: int = Field(
        default=256, description="Patch size for large image scanning"
    )
    patch_stride: int = Field(
        default=128, description="Stride between patches for overlap"
    )
    patch_aggregation: str = Field(
        default="weighted",
        description="Method to aggregate patch results: 'max', 'avg', or 'weighted'",
    )


class TransactionScanRequest(BaseModel):
    transaction_id: str = Field(
        ..., pattern=r"^[a-fA-F0-9]{64}$", description="64-character hex transaction ID"
    )
    extract_images: bool = Field(
        default=True, description="Extract images from transaction"
    )
    scan_options: ScanOptions = Field(
        default_factory=ScanOptions, description="Scanning options"
    )


class ScanResult(BaseModel):
    is_stego: bool = Field(..., description="Whether steganography was detected")
    stego_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Steganography probability"
    )
    stego_type: Optional[str] = Field(
        None, description="Type of steganography detected"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    prediction: str = Field(
        ..., pattern="^(stego|clean)$", description="Model prediction"
    )
    method_id: Optional[int] = Field(None, description="Steganography method ID")
    extracted_message: Optional[str] = Field(
        None, description="Extracted hidden message"
    )
    extraction_error: Optional[str] = Field(
        None, description="Error during message extraction"
    )


class ImageScanResult(BaseModel):
    index: int = Field(..., description="Image index in transaction")
    size_bytes: int = Field(..., ge=0, description="Image size in bytes")
    format: str = Field(..., description="Image format")
    scan_result: ScanResult = Field(..., description="Scanning result")


class TransactionScanResponse(BaseModel):
    transaction_id: str = Field(..., description="Transaction ID")
    block_height: int = Field(..., description="Block height")
    timestamp: str = Field(..., description="Transaction timestamp")
    scan_results: Dict[str, Any] = Field(..., description="Summary of scan results")
    images: List[ImageScanResult] = Field(
        ..., description="Individual image scan results"
    )
    request_id: str = Field(..., description="Unique request ID")


class DirectImageScanResponse(BaseModel):
    scan_result: ScanResult = Field(..., description="Scanning result")
    image_info: Dict[str, Any] = Field(..., description="Image information")
    processing_time_ms: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    request_id: str = Field(..., description="Unique request ID")


class BatchItem(BaseModel):
    type: str = Field(..., pattern="^(transaction|image)$", description="Item type")
    transaction_id: Optional[str] = Field(
        None, description="Transaction ID for transaction type"
    )
    image_data: Optional[str] = Field(
        None, description="Base64 image data for image type"
    )


class BatchScanRequest(BaseModel):
    items: List[BatchItem]
    scan_options: ScanOptions = Field(default_factory=ScanOptions)


class BatchItemResult(BaseModel):
    item_id: str = Field(..., description="Item identifier")
    type: str = Field(..., description="Item type")
    status: str = Field(
        ..., pattern="^(completed|failed)$", description="Processing status"
    )
    stego_detected: bool = Field(..., description="Whether steganography was detected")
    images_with_stego: int = Field(
        ..., ge=0, description="Number of images with steganography"
    )
    total_images: int = Field(..., ge=0, description="Total number of images")
    error: Optional[str] = Field(None, description="Error message if failed")


class BatchScanResponse(BaseModel):
    batch_id: str = Field(..., description="Batch processing ID")
    total_items: int = Field(..., description="Total items in batch")
    processed_items: int = Field(..., description="Successfully processed items")
    stego_detected: int = Field(..., description="Items with steganography detected")
    processing_time_ms: float = Field(..., ge=0, description="Total processing time")
    results: List[BatchItemResult] = Field(..., description="Individual item results")
    request_id: str = Field(..., description="Unique request ID")


class ExtractionResult(BaseModel):
    message_found: bool = Field(..., description="Whether a message was found")
    message: Optional[str] = Field(None, description="Extracted message")
    method_used: Optional[str] = Field(None, description="Steganography method used")
    method_confidence: Optional[float] = Field(
        None, description="Confidence in method detection"
    )
    extraction_details: Dict[str, Any] = Field(..., description="Extraction details")


class ExtractResponse(BaseModel):
    extraction_result: ExtractionResult = Field(..., description="Extraction result")
    image_info: Dict[str, Any] = Field(..., description="Image information")
    processing_time_ms: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    request_id: str = Field(..., description="Unique request ID")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    scanner: Dict[str, Any] = Field(..., description="Scanner status")
    bitcoin: Dict[str, Any] = Field(..., description="Bitcoin node status")


class InfoResponse(BaseModel):
    name: str = Field(..., description="API name")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="API description")
    supported_formats: List[str] = Field(..., description="Supported image formats")
    stego_methods: List[str] = Field(..., description="Supported steganography methods")
    max_image_size: int = Field(..., description="Maximum image size in bytes")
    endpoints: Dict[str, str] = Field(..., description="Available endpoints")


class InscribeResponse(BaseModel):
    request_id: str = Field(..., description="Unique request ID")
    method: str = Field(..., description="Embedding method used")
    message_length: int = Field(..., description="Length of embedded message in bytes")
    output_file: str = Field(..., description="Relative path to saved inscribed image")
    image_bytes: int = Field(..., description="Size of inscribed image in bytes")
    image_sha256: str = Field(
        ..., description="SHA256 hash of inscribed image for ID tracking"
    )
    image_base64: str = Field(..., description="Base64-encoded stego-processed image")
    status: str = Field(..., description="Inscribe status (pending upload)")
    note: str = Field(..., description="Next step hint for Stargate uploader")


class BlockScanRequest(BaseModel):
    block_height: int = Field(..., ge=0, description="Block height to scan")
    scan_options: ScanOptions = Field(
        default_factory=ScanOptions, description="Scanning options"
    )


class BlockScanInscription(BaseModel):
    tx_id: str = Field(..., description="Transaction ID")
    input_index: int = Field(..., description="Input index")
    content_type: str = Field(..., description="Content type")
    content: str = Field(..., description="Content")
    size_bytes: int = Field(..., ge=0, description="Size in bytes")
    file_name: str = Field(..., description="File name")
    file_path: str = Field(..., description="File path")
    scan_result: Optional[ScanResult] = Field(None, description="Scan result")


class BlockScanResponse(BaseModel):
    block_height: int = Field(..., description="Block height")
    block_hash: str = Field(..., description="Block hash")
    timestamp: int = Field(..., description="Block timestamp")
    total_inscriptions: int = Field(..., description="Total inscriptions")
    images_scanned: int = Field(..., description="Images scanned")
    stego_detected: int = Field(..., description="Steganography detected")
    processing_time_ms: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    inscriptions: List[BlockScanInscription] = Field(
        ..., description="Inscription scan results"
    )
    request_id: str = Field(..., description="Unique request ID")


# Global instances
bitcoin_client = BitcoinNodeClient()
scanner_instance = None

# Metrics
REQUEST_COUNT = Counter(
    "starlight_request_total",
    "Total requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "starlight_request_duration_seconds",
    "Request latency seconds",
    ["endpoint"],
)

# Initialize scanner if available
if SCANNER_AVAILABLE:
    try:
        from scanner import StarlightScanner

        # SECURITY: Validate model path is within expected directories
        ALLOWED_MODEL_BASES = ["models", "./models", "../models", "../../models"]
        model_rel_path = "detector_balanced.pth"
        model_base = validate_safe_path("models", ALLOWED_MODEL_BASES)
        model_path = os.path.join(model_base, model_rel_path)
        
        if os.path.exists(model_path):
            scanner_instance = StarlightScanner(model_path, num_workers=4, quiet=True)
            logger.info(f"Scanner initialized with model: {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}")
    except Exception as e:
        logger.error(f"Failed to initialize scanner: {e}")


# Background task for async scanning
async def scan_image_async(
    image_data: bytes, options: ScanOptions, request_id: str
) -> ScanResult:
    """Background task for image scanning"""
    try:
        # Save image data to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(image_data)
            temp_path = temp_file.name

        try:
            if scanner_instance is not None:
                # Use Starlight scanner
                result = scanner_instance.scan_file(temp_path)

                # Convert to ScanResult format
                scan_result = ScanResult(
                    is_stego=result.get("is_stego", False),
                    stego_probability=result.get("stego_probability", 0.0),
                    stego_type=result.get("stego_type"),
                    confidence=result.get("confidence", 0.0),
                    prediction="stego" if result.get("is_stego") else "clean",
                    method_id=result.get("method_id"),
                    extracted_message=(
                        result.get("extracted_message")
                        if options.extract_message
                        else None
                    ),
                    extraction_error=result.get("extraction_error"),
                )

                # Apply confidence threshold
                if scan_result.stego_probability < options.confidence_threshold:
                    scan_result.is_stego = False
                    scan_result.prediction = "clean"

                return scan_result
            else:
                # Stub response when scanner not available
                return ScanResult(
                    is_stego=False,
                    stego_probability=0.1,
                    stego_type=None,
                    confidence=0.9,
                    prediction="clean",
                    method_id=None,
                    extracted_message=None,
                    extraction_error=None,
                )
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass

    except Exception as e:
        logger.error(f"Error scanning image for request {request_id}: {e}")
        return ScanResult(
            is_stego=False,
            stego_probability=0.0,
            stego_type=None,
            confidence=0.0,
            prediction="clean",
            method_id=None,
            extracted_message=None,
            extraction_error=str(e),
        )


def embed_message_to_image(
    image_bytes: bytes, method: str, message: str
) -> tuple[bytes, str]:
    """Embed a UTF-8 message into an image using the selected stego method."""
    if not STEGO_HELPERS_AVAILABLE:
        raise RuntimeError("Stego helpers are unavailable; cannot inscribe message.")
    method_map = {
        "alpha": embed_alpha,
        "lsb": embed_lsb,
        "palette": embed_palette,
        "exif": embed_exif,
        "eoi": embed_eoi,
    }
    method_key = method.lower()
    if method_key not in method_map:
        raise ValueError(
            f"Unsupported method '{method}'. Supported: {', '.join(sorted(method_map.keys()))}"
        )
    cover = Image.open(io.BytesIO(image_bytes))
    embed_func = method_map[method_key]
    if embed_func is None:
        raise RuntimeError(f"Embed function for method '{method}' is not available")
    stego_img = embed_func(cover, message.encode("utf-8"))
    if method_key in ("exif", "eoi"):
        output_format = "JPEG"
    elif method_key == "palette":
        output_format = "GIF"
    else:
        output_format = "PNG"
    buf = io.BytesIO()
    save_kwargs = {}
    if output_format == "JPEG":
        save_kwargs["quality"] = 95
        exif_bytes = stego_img.info.get("exif_bytes")
        if exif_bytes:
            save_kwargs["exif"] = exif_bytes
    stego_img.save(buf, format=output_format, **save_kwargs)
    payload = buf.getvalue()
    if method_key == "eoi":
        append_data = stego_img.info.get("eoi_append")
        if append_data:
            payload += append_data
    return payload, output_format


# Dependency for API key authentication
async def verify_api_key(authorization: str = Header(None)):
    """Verify API key for protected endpoints"""
    required_key = os.environ.get("STARGATE_API_KEY", "demo-api-key")

    if authorization is None:
        # Allow missing header if explicit bypass is configured
        if os.environ.get("ALLOW_ANONYMOUS_SCAN", "false").lower() == "true":
            return "anonymous"
        raise HTTPException(status_code=401, detail="Authorization header required")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")

    api_key = authorization.split(" ")[1]
    # Simple validation (replace with proper JWT verification in production)
    if api_key != required_key:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return api_key


# Import agent manager
AGENT_MANAGER_AVAILABLE = False
from typing import Callable, Dict, Any, Optional

# Type hints for agent manager functions - these will be properly typed after import
get_manager: Any = None
start_agents: Any = None  
stop_agents: Any = None
get_agent_status: Any = None
process_cycle: Any = None

try:
    from starlight.agents.agent_manager import (
        get_manager as _get_manager, 
        start_agents as _start_agents, 
        stop_agents as _stop_agents, 
        get_agent_status as _get_agent_status,
        process_cycle as _process_cycle
    )
    AGENT_MANAGER_AVAILABLE = True
    get_manager = _get_manager
    start_agents = _start_agents
    stop_agents = _stop_agents
    get_agent_status = _get_agent_status
    process_cycle = _process_cycle
    logger.info("Agent manager module loaded successfully")
except ImportError as e:
    AGENT_MANAGER_AVAILABLE = False
    logger.error(f"Could not import agent manager: {e}")

# Global Agent State (backward compatibility)
agent_running = False

def run_agents_loop():
    """Backward compatibility wrapper using new agentManager."""
    global agent_running
    
    if AGENT_MANAGER_AVAILABLE and get_manager is not None:
        # Use the new modular agentManager
        manager = get_manager()
        if manager and hasattr(manager, 'initialize') and manager.initialize():
            agent_running = manager.start(blocking=True)
            logger.info("Agent loop started using AgentManager")
            return agent_running
    
    # Fallback to original logic if manager unavailable
    agent_running = False
    logger.info("Starting Autonomous Agents Loop (fallback mode)")
    try:
        client = StargateClient()
        
        # Bind wallet if configured (needed for write operations like approval)
        if AgentConfig.DONATION_ADDRESS:
            logger.info(f"Attempting to bind wallet: {AgentConfig.DONATION_ADDRESS}")
            client.bind_wallet(AgentConfig.DONATION_ADDRESS)
        else:
            logger.warning("No DONATION_ADDRESS configured. Write operations might fail.")
        
        watcher = WatcherAgent(client, ai_identifier=AgentConfig.AI_IDENTIFIER)
        worker = WorkerAgent(client, ai_identifier=AgentConfig.AI_IDENTIFIER)
        
        while agent_running:
            try:
                # 1. Worker looks for wishes and creates proposals
                logger.info("Agent loop iteration: Starting worker.process_wishes()")
                worker.process_wishes()

                # 2. Watcher looks for proposals (audits/approves them) and tasks
                logger.info("Agent loop iteration: Starting watcher.run_once()")
                tasks = watcher.run_once()
                logger.info(f"Agent loop iteration: Found {len(tasks)} available tasks")
                
                # 3. Worker processes available tasks
                for i, task in enumerate(tasks):
                    logger.info(f"Agent loop iteration: Processing task {i+1}/{len(tasks)}")
                    worker.process_task(task)
                
                # 3. Wait
                # Use a loop for sleep to allow faster shutdown
                for _ in range(AgentConfig.POLL_INTERVAL):
                    if not agent_running:
                        break
                    time.sleep(1)
                
                # Clear any stale flags that might prevent next iteration
                if agent_running:
                    logger.info("Agent loop iteration completed, starting next cycle")
            except Exception as e:
                logger.error(f"Agent loop error: {e}")
                import traceback
                logger.error(f"Agent loop traceback: {traceback.format_exc()}")
                time.sleep(5)
    except Exception as e:
         logger.critical(f"Failed to initialize agents: {e}")

# FastAPI application
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starlight Bitcoin Scanning API starting up...")
    
    # Start Autonomous Agents
    global agent_running, agent_thread
    agent_running = True
    agent_thread = threading.Thread(target=run_agents_loop, daemon=True)
    agent_thread.start()
    logger.info("Autonomous agents thread started")

    yield
    
    # Shutdown
    logger.info("Starlight Bitcoin Scanning API shutting down...")
    agent_running = False
    if agent_thread:
        agent_thread.join(timeout=5)
        logger.info("Autonomous agents thread stopped")


app = FastAPI(
    title="Starlight Bitcoin Scanning API",
    description="API for scanning Bitcoin transactions for steganography",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = datetime.utcnow()
    response = await call_next(request)
    endpoint = request.url.path
    status = getattr(response, "status_code", 500)
    try:
        REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(
            (datetime.utcnow() - start).total_seconds()
        )
    except Exception as e:
        logger.debug(f"metrics middleware skipped: {e}")
    return response


# Health endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Service health check with scanner and Bitcoin node status"""
    scanner_status = {
        "model_loaded": scanner_instance is not None,
        "model_version": "v4-prod" if scanner_instance else "none",
        "model_path": "models/detector_balanced.pth",  # Relative path for security
        "device": "cpu",  # Would detect actual device
    }

    bitcoin_status = {
        "node_connected": bitcoin_client.connected,
        "node_url": bitcoin_client.node_url,
        "block_height": await bitcoin_client.get_block_height(),
    }

    return HealthResponse(
        status=(
            "healthy" if scanner_instance and bitcoin_client.connected else "degraded"
        ),
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        scanner=scanner_status,
        bitcoin=bitcoin_status,
    )


@app.get("/info", response_model=InfoResponse, tags=["Info"])
async def api_info():
    """API information and capabilities"""
    return InfoResponse(
        name="Starlight Bitcoin Steganography Scanner",
        version="1.0.0",
        description="AI-powered steganography detection for Bitcoin transaction images",
        supported_formats=["png", "jpg", "jpeg", "gif", "bmp", "webp"],
        stego_methods=["alpha", "palette", "lsb.rgb", "exif", "raw"],
        max_image_size=10485760,  # 10MB
        endpoints={
            "scan_tx": "/scan/transaction",
            "scan_image": "/scan/image",
            "batch_scan": "/scan/batch",
            "extract": "/extract",
            "inscribe": "/inscribe",
        },
    )


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Transaction scanning endpoint
@app.post(
    "/scan/transaction", response_model=TransactionScanResponse, tags=["Scanning"]
)
async def scan_transaction(
    request: TransactionScanRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """Scan a Bitcoin transaction for steganography in embedded images"""
    start_time = datetime.utcnow()
    request_id = str(uuid.uuid4())

    try:
        # Get transaction data
        tx_data = await bitcoin_client.get_transaction(request.transaction_id)

        # Extract images from transaction
        images = []
        if request.extract_images:
            extracted_images = await bitcoin_client.extract_images(
                request.transaction_id
            )

            for img_data in extracted_images:
                # Scan image asynchronously
                scan_result = await scan_image_async(
                    img_data["data"], request.scan_options, request_id
                )

                images.append(
                    ImageScanResult(
                        index=img_data["index"],
                        size_bytes=img_data["size_bytes"],
                        format=img_data["format"],
                        scan_result=scan_result,
                    )
                )

        # Calculate summary
        stego_detected = any(img.scan_result.is_stego for img in images)
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        scan_summary = {
            "images_found": len(images),
            "images_scanned": len(images),
            "stego_detected": stego_detected,
            "processing_time_ms": processing_time,
        }

        return TransactionScanResponse(
            transaction_id=request.transaction_id,
            block_height=tx_data["block_height"],
            timestamp=tx_data["timestamp"],
            scan_results=scan_summary,
            images=images,
            request_id=request_id,
        )

    except Exception as e:
        logger.error(f"Error scanning transaction {request.transaction_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Transaction scan failed: {str(e)}"
        )


# Direct image scanning endpoint
@app.post("/scan/image", response_model=DirectImageScanResponse, tags=["Scanning"])
async def scan_image(
    image: UploadFile = File(...),
    extract_message: bool = Form(True),
    confidence_threshold: float = Form(0.5),
    include_metadata: bool = Form(True),
    api_key: str = Depends(verify_api_key),
):
    """Scan a directly uploaded image for steganography"""
    start_time = datetime.utcnow()
    request_id = str(uuid.uuid4())

    try:
        # Read image data
        image_data = await image.read()

        # Validate image size
        if len(image_data) > 10485760:  # 10MB limit
            raise HTTPException(status_code=413, detail="Image too large")

        # Create scan options
        options = ScanOptions(
            extract_message=extract_message,
            confidence_threshold=confidence_threshold,
            include_metadata=include_metadata,
        )

        # Scan image
        scan_result = await scan_image_async(image_data, options, request_id)

        # Get image info
        filename = image.filename or "unknown"
        image_info = {
            "filename": filename,
            "size_bytes": len(image_data),
            "format": filename.split(".")[-1].lower() if "." in filename else "unknown",
        }

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return DirectImageScanResponse(
            scan_result=scan_result,
            image_info=image_info,
            processing_time_ms=processing_time,
            request_id=request_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scanning image: {e}")
        raise HTTPException(status_code=500, detail=f"Image scan failed: {str(e)}")


# Inscription preparation endpoint (embed text into image and save for Stargate upload)
@app.post("/inscribe", response_model=InscribeResponse, tags=["Inscribe"])
async def inscribe_image(
    image: UploadFile = File(...),
    message: str = Form(...),
    method: str = Form("alpha"),
    api_key: str = Depends(verify_api_key),
):
    """Embed a text message into an image and save it for Stargate to broadcast."""
    request_id = str(uuid.uuid4())
    try:
        if not message:
            raise HTTPException(
                status_code=400, detail="Message is required for inscription"
            )
        image_bytes = await image.read()
        if len(image_bytes) > 10485760:
            raise HTTPException(status_code=413, detail="Image too large")
        try:
            stego_bytes, output_format = embed_message_to_image(
                image_bytes, method, message
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except RuntimeError as re:
            raise HTTPException(status_code=503, detail=str(re))

        filename = image.filename or "inscribe.png"
        safe_name = os.path.basename(filename) or "inscribe.png"
        if output_format:
            ext_map = {"JPEG": ".jpg", "PNG": ".png", "GIF": ".gif"}
            new_ext = ext_map.get(output_format)
            if new_ext:
                base, _ = os.path.splitext(safe_name)
                safe_name = base + new_ext

        # Optional direct handoff to Stargate via REST
        ingest_url = os.environ.get("STARGATE_INGEST_URL")
        ingest_token = os.environ.get("STARGATE_INGEST_TOKEN", "")
        ingest_result = None
        if ingest_url:
            try:
                payload = {
                    "id": request_id,
                    "filename": safe_name,
                    "method": method,
                    "message_length": len(message.encode("utf-8")),
                    "image_base64": base64.b64encode(stego_bytes).decode("utf-8"),
                    # Include the embedded message so Stargate can surface it in pending UI
                    "metadata": {"embedded_message": message},
                }
                headers = {"Content-Type": "application/json"}
                if ingest_token:
                    headers["X-Ingest-Token"] = ingest_token
                resp = requests.post(
                    ingest_url, json=payload, headers=headers, timeout=10
                )
                ingest_result = {"status_code": resp.status_code, "response": resp.text}
            except Exception as e:
                ingest_result = {"error": str(e)}

        # Calculate SHA256 hash of the inscribed image
        image_sha256 = hashlib.sha256(stego_bytes).hexdigest()

        response_payload = {
            "request_id": request_id,
            "method": method,
            "message_length": len(message.encode("utf-8")),
            "output_file": safe_name,
            "image_bytes": len(stego_bytes),
            "image_sha256": image_sha256,
            "image_base64": base64.b64encode(stego_bytes).decode("utf-8"),
            "status": "ingested" if ingest_result else "pending_upload",
            "note": (
                "Ingested to Stargate via REST"
                if ingest_result
                else "No ingest URL configured"
            ),
            "ingest": ingest_result,
        }
        return InscribeResponse(**response_payload)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error embedding inscription: {e}")
        raise HTTPException(status_code=500, detail=f"Inscribe failed: {str(e)}")


# Batch scanning endpoint
@app.post("/scan/batch", response_model=BatchScanResponse, tags=["Scanning"])
async def scan_batch(
    request: BatchScanRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """Scan multiple transactions or images in a batch"""
    start_time = datetime.utcnow()
    request_id = str(uuid.uuid4())
    batch_id = f"batch_{uuid.uuid4().hex[:8]}"

    try:
        results = []
        stego_count = 0

        for item in request.items:
            try:
                if item.type == "transaction":
                    # Scan transaction
                    if item.transaction_id is None:
                        raise ValueError(
                            "Transaction ID is required for transaction type items"
                        )
                    tx_data = await bitcoin_client.get_transaction(item.transaction_id)
                    images = await bitcoin_client.extract_images(item.transaction_id)

                    item_stego_count = 0
                    for img_data in images:
                        scan_result = await scan_image_async(
                            img_data["data"], request.scan_options, request_id
                        )
                        if scan_result.is_stego:
                            item_stego_count += 1

                    if item_stego_count > 0:
                        stego_count += 1

                    results.append(
                        BatchItemResult(
                            item_id=item.transaction_id or "unknown",
                            type="transaction",
                            status="completed",
                            stego_detected=item_stego_count > 0,
                            images_with_stego=item_stego_count,
                            total_images=len(images),
                            error=None,
                        )
                    )

                elif item.type == "image":
                    # Scan base64 image
                    if item.image_data is None:
                        raise ValueError("Image data is required for image type items")
                    image_data = base64.b64decode(item.image_data)
                    scan_result = await scan_image_async(
                        image_data, request.scan_options, request_id
                    )

                    if scan_result.is_stego:
                        stego_count += 1

                    results.append(
                        BatchItemResult(
                            item_id=f"image_{len(results)}",
                            type="image",
                            status="completed",
                            stego_detected=scan_result.is_stego,
                            images_with_stego=1 if scan_result.is_stego else 0,
                            total_images=1,
                            error=None,
                        )
                    )

            except Exception as e:
                results.append(
                    BatchItemResult(
                        item_id=item.transaction_id or f"image_{len(results)}",
                        type=item.type,
                        status="failed",
                        stego_detected=False,
                        images_with_stego=0,
                        total_images=0,
                        error=str(e),
                    )
                )

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return BatchScanResponse(
            batch_id=batch_id,
            total_items=len(request.items),
            processed_items=len([r for r in results if r.status == "completed"]),
            stego_detected=stego_count,
            processing_time_ms=processing_time,
            results=results,
            request_id=request_id,
        )

    except Exception as e:
        logger.error(f"Error in batch scan: {e}")
        raise HTTPException(status_code=500, detail=f"Batch scan failed: {str(e)}")


# Message extraction endpoint
@app.post("/extract", response_model=ExtractResponse, tags=["Extraction"])
async def extract_message(
    image: UploadFile = File(...),
    method: Optional[str] = Form(None),
    force_extract: bool = Form(False),
    api_key: str = Depends(verify_api_key),
):
    """Extract hidden messages from a steganographic image"""
    start_time = datetime.utcnow()
    request_id = str(uuid.uuid4())

    try:
        # Read image data
        image_data = await image.read()

        # Save to temporary file for extraction
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(image_data)
            temp_path = temp_file.name

        try:
            if scanner_instance:
                # Use scanner to extract message
                result = scanner_instance.scan_file(temp_path)

                extraction_result = ExtractionResult(
                    message_found=bool(result.get("extracted_message")),
                    message=result.get("extracted_message"),
                    method_used=result.get("stego_type"),
                    method_confidence=result.get("confidence"),
                    extraction_details={
                        "bits_extracted": (
                            len(result.get("extracted_message", "")) * 8
                            if result.get("extracted_message")
                            else 0
                        ),
                        "encoding": "utf-8",
                        "corruption_detected": False,
                    },
                )
            else:
                # Stub response
                extraction_result = ExtractionResult(
                    message_found=False,
                    message=None,
                    method_used=None,
                    method_confidence=None,
                    extraction_details={
                        "bits_extracted": 0,
                        "encoding": "utf-8",
                        "corruption_detected": False,
                    },
                )

            filename = image.filename or "unknown"
            image_info = {
                "filename": filename,
                "size_bytes": len(image_data),
                "format": (
                    filename.split(".")[-1].lower() if "." in filename else "unknown"
                ),
            }

            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return ExtractResponse(
                extraction_result=extraction_result,
                image_info=image_info,
                processing_time_ms=processing_time,
                request_id=request_id,
            )

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass

    except Exception as e:
        logger.error(f"Error extracting message: {e}")
        raise HTTPException(
            status_code=500, detail=f"Message extraction failed: {str(e)}"
        )


# Transaction lookup endpoint
@app.get("/transaction/{txid}", tags=["Transaction"])
async def get_transaction(
    txid: str,
    include_images: bool = Query(False),
    image_format: str = Query("info"),
    api_key: str = Depends(verify_api_key),
):
    """Get transaction details and available images"""
    try:
        # Validate transaction ID
        if not len(txid) == 64 or not all(c in "0123456789abcdefABCDEF" for c in txid):
            raise HTTPException(status_code=400, detail="Invalid transaction ID format")

        # Get transaction data
        tx_data = await bitcoin_client.get_transaction(txid)

        response_data = {
            "transaction_id": txid,
            "block_height": tx_data["block_height"],
            "timestamp": tx_data["timestamp"],
            "status": tx_data["status"],
            "total_images": 0,
        }

        if include_images:
            images = await bitcoin_client.extract_images(txid)
            response_data["images"] = []

            for img in images:
                img_data = {
                    "index": img["index"],
                    "size_bytes": img["size_bytes"],
                    "format": img["format"],
                }

                if image_format == "base64":
                    img_data["data"] = base64.b64encode(img["data"]).decode("utf-8")
                elif image_format == "info":
                    # Just info, no data
                    pass

                response_data["images"].append(img_data)

            response_data["total_images"] = len(images)

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting transaction {txid}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Transaction lookup failed: {str(e)}"
        )


# Error handlers
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


# Block scanning endpoint
@app.post("/scan/block", response_model=BlockScanResponse, tags=["Scanning"])
async def scan_block(
    request: BlockScanRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """Scan a Bitcoin block for steganography in all inscriptions"""
    start_time = datetime.utcnow()
    request_id = str(uuid.uuid4())
    blocks_dir = os.environ.get("BLOCKS_DIR") or globals().get("BLOCKS_DIR", "blocks")

    try:
        # Find block directory (hash suffix unknown; glob for match)
        pattern = os.path.join(blocks_dir, f"{request.block_height}_*")
        matches = glob.glob(pattern)
        if not matches:
            raise HTTPException(
                status_code=404, detail=f"Block {request.block_height} not found"
            )
        block_dir = matches[0]

        # Load block data
        block_json_path = os.path.join(block_dir, "block.json")
        block_data = {}
        if os.path.exists(block_json_path):
            with open(block_json_path, "r") as f:
                block_data = json.load(f)

        # Load existing inscriptions if available
        inscriptions_json_path = os.path.join(block_dir, "inscriptions.json")
        existing_inscriptions = {}
        if os.path.exists(inscriptions_json_path):
            with open(inscriptions_json_path, "r") as f:
                existing_inscriptions = json.load(f)

        # Scan images in block
        images_dir = os.path.join(block_dir, "images")
        scanned_inscriptions = []
        images_scanned = 0
        stego_detected = 0

        if os.path.exists(images_dir):
            image_files = [
                f
                for f in os.listdir(images_dir)
                if f.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
                )
            ]

            for image_file in image_files:
                image_path = os.path.join(images_dir, image_file)
                try:
                    # Scan image
                    scan_result = await scan_image_async(
                        open(image_path, "rb").read(), request.scan_options, request_id
                    )

                    # Get inscription info
                    inscription_info = None
                    for insc in existing_inscriptions.get("inscriptions", []):
                        if insc.get("file_name") == image_file:
                            inscription_info = insc
                            break

                    if not inscription_info:
                        # Create basic inscription info
                        inscription_info = {
                            "tx_id": f"unknown_{image_file}",
                            "input_index": 0,
                            "content_type": f"image/{image_file.split('.')[-1]}",
                            "content": f"Image file: {image_file}",
                            "size_bytes": os.path.getsize(image_path),
                            "file_name": image_file,
                            "file_path": f"images/{image_file}",
                        }

                    # Add scan result to inscription
                    scan_payload = scan_result.dict()
                    inscription_info["scan_result"] = scan_payload
                    if scan_payload.get("is_stego"):
                        stego_detected += 1

                    callback_payload = {
                        "request_id": request_id,
                        "block_height": request.block_height,
                        "tx_id": inscription_info.get("tx_id"),
                        "file_name": inscription_info.get("file_name"),
                        "content_type": inscription_info.get("content_type"),
                        "size_bytes": inscription_info.get("size_bytes"),
                        "scan_result": scan_payload,
                        "metadata": {
                            "scanner": "starlight-api",
                            "scanned_at": int(time.time()),
                            "source": "scan_block",
                        },
                    }
                    callback_result = send_stargate_callback(callback_payload)
                    if callback_result and not callback_result.get("skipped"):
                        inscription_info["stargate_callback"] = callback_result

                    scanned_inscriptions.append(inscription_info)
                    images_scanned += 1

                except Exception as e:
                    logger.error(f"Error scanning image {image_file}: {e}")
                    # Add inscription without scan result
                    scanned_inscriptions.append(
                        {
                            "tx_id": f"unknown_{image_file}",
                            "input_index": 0,
                            "content_type": f"image/{image_file.split('.')[-1]}",
                            "content": f"Image file: {image_file}",
                            "size_bytes": os.path.getsize(image_path),
                            "file_name": image_file,
                            "file_path": f"images/{image_file}",
                            "scan_result": None,
                            "error": str(e),
                        }
                    )
                    images_scanned += 1

        # Update block data with scan results
        updated_block_data = block_data.copy()
        updated_block_data["stego_detected"] = stego_detected
        updated_block_data["images_scanned"] = images_scanned
        updated_block_data["scan_timestamp"] = int(start_time.timestamp())

        # Save updated block data
        with open(block_json_path, "w") as f:
            json.dump(updated_block_data, f, indent=2)

        # Update inscriptions with scan results
        updated_inscriptions = existing_inscriptions.copy()
        updated_inscriptions["inscriptions"] = scanned_inscriptions

        with open(inscriptions_json_path, "w") as f:
            json.dump(updated_inscriptions, f, indent=2)

        # Update global inscriptions JSON
        global_inscriptions_path = os.path.join(blocks_dir, "global_inscriptions.json")
        global_inscriptions = {}

        if os.path.exists(global_inscriptions_path):
            with open(global_inscriptions_path, "r") as f:
                global_inscriptions = json.load(f)

        block_key = f"{request.block_height}_{block_data.get('hash', 'unknown')}"
        global_inscriptions[block_key] = {
            "hash": block_data.get("hash", "unknown"),
            "height": request.block_height,
            "image_count": images_scanned,
            "inscriptions": scanned_inscriptions,
            "timestamp": block_data.get("timestamp", 0),
            "stego_detected": stego_detected,
            "scan_timestamp": int(start_time.timestamp()),
        }

        with open(global_inscriptions_path, "w") as f:
            json.dump(global_inscriptions, f, indent=2)

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        block_hash = (
            block_data.get("block_hash")
            or block_data.get("BlockHash")
            or block_data.get("block_header", {}).get("Hash")
            or block_data.get("hash")
            or os.path.basename(block_dir).split("_")[1]
        )
        block_timestamp = block_data.get("timestamp", 0)

        return BlockScanResponse(
            block_height=request.block_height,
            block_hash=block_hash or "unknown",
            timestamp=block_timestamp,
            total_inscriptions=len(scanned_inscriptions),
            images_scanned=images_scanned,
            stego_detected=stego_detected,
            processing_time_ms=processing_time,
            inscriptions=scanned_inscriptions,
            request_id=request_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scanning block {request.block_height}: {e}")
        raise HTTPException(status_code=500, detail=f"Block scan failed: {str(e)}")


# Legacy agent status endpoint (backward compatibility)
@app.get("/agents/legacy/status", tags=["Agents"])
async def get_legacy_agent_status():
    """Get the status of the legacy autonomous agents"""
    return {
        "running": agent_running,
        "manager_available": AGENT_MANAGER_AVAILABLE,
        "config": {
            "stargate_url": AgentConfig.STARGATE_API_URL,
            "ai_identifier": AgentConfig.AI_IDENTIFIER,
            "poll_interval": AgentConfig.POLL_INTERVAL
        }
    }

# Updated status endpoint
@app.get("/agents/status", tags=["Agents"])
async def get_agents_status():
    """Get the status of the autonomous agents"""
    if AGENT_MANAGER_AVAILABLE and get_agent_status:
        # Get status from the new AgentManager
        manager_status = get_agent_status()
        return {
            "running": agent_running,
            "manager_available": True,
            "config": {
                "stargate_url": AgentConfig.STARGATE_API_URL,
                "ai_identifier": AgentConfig.AI_IDENTIFIER,
                "poll_interval": AgentConfig.POLL_INTERVAL
            },
            "internal_status": manager_status
        }
    else:
        # Fallback to legacy status
        return {
            "running": agent_running,
            "manager_available": False,
            "config": {
                "stargate_url": AgentConfig.STARGATE_API_URL,
                "ai_identifier": AgentConfig.AI_IDENTIFIER,
                "poll_interval": AgentConfig.POLL_INTERVAL
            }
        }


 # Removed agent control endpoints for security
# Agent control functionality is now internal-only to prevent external manipulation

# Dynamic Loading for Self-Improvement
class SandboxLoadRequest(BaseModel):
    visible_pixel_hash: str = Field(..., description="Contract hash for isolation")
    function_name: str = Field(..., description="Function to load")
    module_name: Optional[str] = Field(None, description="Module containing function")
    function_type: str = Field(..., pattern="^(scan|agent|api)$", description="Type of function")

class SandboxLoadResponse(BaseModel):
    status: str = Field(..., description="Load status")
    function_loaded: bool = Field(..., description="Whether function was successfully loaded")
    endpoint_path: Optional[str] = Field(None, description="Dynamic endpoint path")
    sandbox_id: str = Field(..., description="Unique sandbox load ID")

@app.post("/sandbox/load", response_model=SandboxLoadResponse, tags=["Sandbox"])
async def load_sandbox_function(
    request: SandboxLoadRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """Load an agent-developed function from sandbox"""
    sandbox_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"Loading sandbox function: {request.function_name} for {request.visible_pixel_hash}")
        
        # Create load request
        load_request = DynamicLoadRequest(
            visible_pixel_hash=request.visible_pixel_hash,
            function_name=request.function_name,
            module_name=request.module_name,
            endpoint_path=f"/sandbox/{request.function_type}/{request.function_name}",
            method="GET"
        )
        
        # Load function using dynamic loader
        loaded_module = dynamic_loader.load_function(load_request)
        
        # Register as endpoint
        func = dynamic_loader.get_function(request.visible_pixel_hash, request.function_name)
        if func:
            # Add dynamic route for sandbox function
            app.add_api_route(
                loaded_module.endpoint_path,
                func,
                methods=[loaded_module.method],
                name=f"sandbox_{loaded_module.module_id[:8]}",
                operation_id=f"sandbox_{request.function_type}_{request.function_name}",
                include_in_schema=True
            )
            
            # Refresh OpenAPI schema
            app.openapi_schema = None
            
            logger.info(f"Sandbox function loaded: {loaded_module.endpoint_path}")
            
            return SandboxLoadResponse(
                status="success",
                function_loaded=True,
                endpoint_path=loaded_module.endpoint_path,
                sandbox_id=sandbox_id
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to load function")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sandbox function loading failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Sandbox function loading failed: {str(e)}"
        )

@app.get("/sandbox/status", tags=["Sandbox"])
async def get_sandbox_status():
    """Get current sandbox capabilities and loaded functions"""
    try:
        loaded_functions = dynamic_loader.list_loaded_functions()
        
        # Categorize functions
        scan_functions = []
        agent_functions = []
        api_functions = []
        
        for module_id, loaded_module in loaded_functions.items():
            function_info = {
                "module_id": module_id,
                "function_name": loaded_module.function_name,
                "endpoint_path": loaded_module.endpoint_path,
                "loaded_at": loaded_module.loaded_at,
                "visible_pixel_hash": loaded_module.visible_pixel_hash
            }
            
            # Categorize by endpoint path
            if "scan" in loaded_module.endpoint_path:
                scan_functions.append(function_info)
            elif "agent" in loaded_module.endpoint_path:
                agent_functions.append(function_info)
            else:
                api_functions.append(function_info)
        
        return {
            "status": "active",
            "total_loaded": len(loaded_functions),
            "scan_functions": scan_functions,
            "agent_functions": agent_functions,
            "api_functions": api_functions,
            "sandbox_enabled": True,
            "agent_development_capable": AGENT_MANAGER_AVAILABLE,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting sandbox status: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Status check failed: {str(e)}"
        )

@app.delete("/sandbox/unload/{visible_pixel_hash}/{function_name}", tags=["Sandbox"])
async def unload_sandbox_function(
    visible_pixel_hash: str,
    function_name: str,
    api_key: str = Depends(verify_api_key),
):
    """Unload a sandbox function"""
    try:
        logger.info(f"Unloading sandbox function: {function_name} for {visible_pixel_hash}")
        
        success = dynamic_loader.unload_function(visible_pixel_hash, function_name)
        
        if success:
            # Refresh OpenAPI schema
            app.openapi_schema = None
            
            return {
                "status": "success",
                "message": f"Sandbox function '{function_name}' unloaded successfully",
                "visible_pixel_hash": visible_pixel_hash
            }
        else:
            raise HTTPException(status_code=404, detail="Function not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sandbox function unloading failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Unload failed: {str(e)}"
        )

@app.post("/sandbox/auto-load", tags=["Sandbox"])
async def trigger_auto_sandbox_load(
    api_key: str = Depends(verify_api_key),
):
    """Trigger autonomous agents to create and load sandbox functions"""
    try:
        logger.info("Triggering auto sandbox load cycle")
        
        if not AGENT_MANAGER_AVAILABLE:
            raise HTTPException(
                status_code=503, 
                detail="Agent manager not available for auto sandbox loading"
            )
        
        # Trigger agent sandbox cycle
        if process_cycle:
            # Process one cycle to generate functions
            sandbox_results = process_cycle()
            
            return {
                "status": "loaded",
                "sandbox_cycle": "completed",
                "results": sandbox_results,
                "message": "Auto sandbox load cycle completed. Check loaded functions.",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=503, 
                detail="Auto sandbox loading not available"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auto sandbox loading failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Auto sandbox loading failed: {str(e)}"
        )

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API root endpoint with basic information"""
    loaded_count = len(dynamic_loader.list_loaded_functions())
    
    return {
        "name": "Starlight Bitcoin Scanning API with Sandbox",
        "version": "2.0.0",
        "description": "API for scanning Bitcoin transactions for steganography with dynamic sandbox loading capabilities",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "scan_transaction": "/scan/transaction",
            "scan_image": "/scan/image",
            "batch_scan": "/scan/batch",
            "scan_block": "/scan/block",
            "extract": "/extract",
            "inscribe": "/inscribe",
            "get_transaction": "/transaction/{txid}",
            "sandbox": {
                "load": "/sandbox/load",
                "status": "/sandbox/status", 
                "unload": "/sandbox/unload/{hash}/{function}",
                "auto_load": "/sandbox/auto-load"
            },
            "agents_status": "/agents/status"
        },
        "documentation": "/docs",
        "agent_docs": {
            "sandbox_guide": "/AGENT_SANDBOX_GUIDE.md",
            "quick_reference": "/AGENT_SANDBOX_QUICK_REF.md",
            "dynamic_api": "/DYNAMIC_API_README.md"
        },
        "timestamp": datetime.utcnow().isoformat(),
        "scanner_available": SCANNER_AVAILABLE,
        "sandbox": {
            "enabled": True,
            "loaded_functions": loaded_count,
            "auto_loading_available": AGENT_MANAGER_AVAILABLE,
            "isolation_based_on": "visible_pixel_hash"
        }
    }


# Main execution for development
if __name__ == "__main__":
    try:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8080)
    except ImportError:
        logger.error("uvicorn not available - cannot run server")
