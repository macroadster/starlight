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
import asyncio
import logging
import base64
import io
import os
import tempfile
import hashlib
import json
from pathlib import Path
import requests
from contextlib import asynccontextmanager
import glob

from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default blocks directory (overridable via env)
BLOCKS_DIR = os.environ.get("BLOCKS_DIR", "blocks")

# Check if FastAPI is available
FASTAPI_AVAILABLE = False
try:
    from fastapi import FastAPI, HTTPException, Depends, Query, Header, BackgroundTasks, File, UploadFile, Form
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    logger.warning("FastAPI not available - running in stub mode")
    
    # Create stub classes for development
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class Field:
        def __init__(self, default=None, **kwargs):
            self.default = default
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class FastAPI:
        def __init__(self, **kwargs):
            self.title = kwargs.get('title', 'Stub API')
            self.version = kwargs.get('version', '1.0.0')
        
        def get(self, path, **kwargs):
            def decorator(func):
                return func
            return decorator
        
        def post(self, path, **kwargs):
            def decorator(func):
                return func
            return decorator
    
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)
    
    def Depends(dependency):
        return dependency
    
    def Query(default=None, **kwargs):
        return default
    
    def Header(default=None):
        return default
    
    class BackgroundTasks:
        def add_task(self, func, *args, **kwargs):
            func(*args, **kwargs)
    
    class File:
        def __init__(self, default=None, **kwargs):
            self.default = default
    
    class UploadFile:
        def __init__(self, filename=None, file=None):
            self.filename = filename
            self.file = file
    
    class Form:
        def __init__(self, default=None, **kwargs):
            self.default = default
    
    class JSONResponse:
        def __init__(self, status_code=200, content=None):
            self.status_code = status_code
            self.content = content or {}

# Import scanner components
try:
    from scanner import StarlightScanner, _scan_logic
    SCANNER_AVAILABLE = True
    logger.info("Starlight scanner loaded successfully")
except ImportError as e:
    SCANNER_AVAILABLE = False
    logger.error(f"Could not import Starlight scanner: {e}")

# Stego embedding helpers (re-use existing tool functions when available)
try:
    from scripts.stego_tool import embed_alpha, embed_lsb, embed_palette, embed_exif, embed_eoi
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
                    "script_pubkey": "OP_RETURN " + "48656c6c6f20576f726c64",  # "Hello World" in hex
                    "value": 0
                }
            ]
        }
    
    async def extract_images(self, tx_id: str) -> List[Dict[str, Any]]:
        """Extract images from transaction outputs"""
        # Stub implementation - would parse actual transaction data
        return [
            {
                "index": 0,
                "data": base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="),
                "format": "png",
                "size_bytes": 67
            }
        ]
    
    async def get_block_height(self) -> int:
        """Get current block height"""
        return 856789

# Pydantic models for request/response validation
class ScanOptions:
    def __init__(self, **kwargs):
        self.extract_message = kwargs.get('extract_message', True)
        self.confidence_threshold = kwargs.get('confidence_threshold', 0.5)
        self.include_metadata = kwargs.get('include_metadata', True)

if FASTAPI_AVAILABLE:
    class ScanOptions(BaseModel):
        extract_message: bool = Field(default=True, description="Extract hidden messages if stego detected")
        confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
        include_metadata: bool = Field(default=True, description="Include detailed metadata in response")
    
    class TransactionScanRequest(BaseModel):
        transaction_id: str = Field(..., pattern=r'^[a-fA-F0-9]{64}$', description="64-character hex transaction ID")
        extract_images: bool = Field(default=True, description="Extract images from transaction")
        scan_options: ScanOptions = Field(default_factory=ScanOptions, description="Scanning options")
    
    class ScanResult(BaseModel):
        is_stego: bool = Field(..., description="Whether steganography was detected")
        stego_probability: float = Field(..., ge=0.0, le=1.0, description="Steganography probability")
        stego_type: Optional[str] = Field(None, description="Type of steganography detected")
        confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
        prediction: str = Field(..., pattern="^(stego|clean)$", description="Model prediction")
        method_id: Optional[int] = Field(None, description="Steganography method ID")
        extracted_message: Optional[str] = Field(None, description="Extracted hidden message")
        extraction_error: Optional[str] = Field(None, description="Error during message extraction")
    
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
        images: List[ImageScanResult] = Field(..., description="Individual image scan results")
        request_id: str = Field(..., description="Unique request ID")
    
    class DirectImageScanResponse(BaseModel):
        scan_result: ScanResult = Field(..., description="Scanning result")
        image_info: Dict[str, Any] = Field(..., description="Image information")
        processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")
        request_id: str = Field(..., description="Unique request ID")
    
    class BatchItem(BaseModel):
        type: str = Field(..., pattern="^(transaction|image)$", description="Item type")
        transaction_id: Optional[str] = Field(None, description="Transaction ID for transaction type")
        image_data: Optional[str] = Field(None, description="Base64 image data for image type")
    
    class BatchScanRequest(BaseModel):
        items: List[BatchItem] = Field(..., min_items=1, max_items=50, description="Items to scan")
        scan_options: ScanOptions = Field(default_factory=ScanOptions, description="Scanning options")
    
    class BatchItemResult(BaseModel):
        item_id: str = Field(..., description="Item identifier")
        type: str = Field(..., description="Item type")
        status: str = Field(..., pattern="^(completed|failed)$", description="Processing status")
        stego_detected: bool = Field(..., description="Whether steganography was detected")
        images_with_stego: int = Field(..., ge=0, description="Number of images with steganography")
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
        method_confidence: Optional[float] = Field(None, description="Confidence in method detection")
        extraction_details: Dict[str, Any] = Field(..., description="Extraction details")
    
    class ExtractResponse(BaseModel):
        extraction_result: ExtractionResult = Field(..., description="Extraction result")
        image_info: Dict[str, Any] = Field(..., description="Image information")
        processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")
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
        status: str = Field(..., description="Inscribe status (pending upload)")
        note: str = Field(..., description="Next step hint for Stargate uploader")
    
    class BlockScanRequest(BaseModel):
        block_height: int = Field(..., ge=0, description="Block height to scan")
        scan_options: ScanOptions = Field(default_factory=ScanOptions, description="Scanning options")
    
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
        processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")
        inscriptions: List[BlockScanInscription] = Field(..., description="Inscription scan results")
        request_id: str = Field(..., description="Unique request ID")

# Global instances
bitcoin_client = BitcoinNodeClient()
scanner_instance = None
INSCRIPTION_OUTBOX = Path(os.environ.get("STARLIGHT_INSCRIPTION_OUTBOX", "inscriptions/pending"))
INSCRIPTION_OUTBOX.mkdir(parents=True, exist_ok=True)

# Initialize scanner if available
if SCANNER_AVAILABLE:
    try:
        model_path = "models/detector_balanced.onnx"
        if os.path.exists(model_path):
            scanner_instance = StarlightScanner(model_path, num_workers=4, quiet=True)
            logger.info(f"Scanner initialized with model: {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}")
    except Exception as e:
        logger.error(f"Failed to initialize scanner: {e}")

# Background task for async scanning
async def scan_image_async(image_data: bytes, options: ScanOptions, request_id: str) -> ScanResult:
    """Background task for image scanning"""
    try:
        # Save image data to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file.write(image_data)
            temp_path = temp_file.name
        
        try:
            if scanner_instance:
                # Use Starlight scanner
                result = scanner_instance.scan_file(temp_path)
                
                 # Convert to ScanResult format
                if FASTAPI_AVAILABLE:
                    scan_result = ScanResult(
                        is_stego=result.get("is_stego", False),
                        stego_probability=result.get("stego_probability", 0.0),
                        stego_type=result.get("stego_type"),
                        confidence=result.get("confidence", 0.0),
                        prediction="stego" if result.get("is_stego") else "clean",
                        method_id=result.get("method_id"),
                        extracted_message=result.get("extracted_message") if options.extract_message else None,
                        extraction_error=result.get("extraction_error")
                    )
                else:
                    scan_result = {
                        "is_stego": result.get("is_stego", False),
                        "stego_probability": result.get("stego_probability", 0.0),
                        "stego_type": result.get("stego_type"),
                        "confidence": result.get("confidence", 0.0),
                        "prediction": "stego" if result.get("is_stego") else "clean",
                        "method_id": result.get("method_id"),
                        "extracted_message": result.get("extracted_message") if options.extract_message else None,
                        "extraction_error": result.get("extraction_error")
                    }
                
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
                    confidence=0.9,
                    prediction="clean"
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
            confidence=0.0,
            prediction="clean",
            extraction_error=str(e)
        )


def embed_message_to_image(image_bytes: bytes, method: str, message: str) -> bytes:
    """Embed a UTF-8 message into an image using the selected stego method."""
    if not STEGO_HELPERS_AVAILABLE:
        raise RuntimeError("Stego helpers are unavailable; cannot inscribe message.")
    method_map = {
        "alpha": embed_alpha,
        "lsb": embed_lsb,
        "palette": embed_palette,
        "exif": embed_exif,
        "eoi": embed_eoi
    }
    method_key = method.lower()
    if method_key not in method_map:
        raise ValueError(f"Unsupported method '{method}'. Supported: {', '.join(sorted(method_map.keys()))}")
    cover = Image.open(io.BytesIO(image_bytes))
    stego_img = method_map[method_key](cover, message.encode("utf-8"))
    buf = io.BytesIO()
    stego_img.save(buf, format="PNG")
    return buf.getvalue()

# Dependency for API key authentication
async def verify_api_key(authorization: str = Header(None)):
    """Verify API key for protected endpoints"""
    if not FASTAPI_AVAILABLE:
        return "demo-api-key"
    
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

# FastAPI application
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starlight Bitcoin Scanning API starting up...")
    yield
    # Shutdown
    logger.info("Starlight Bitcoin Scanning API shutting down...")

app = FastAPI(
    title="Starlight Bitcoin Scanning API",
    description="API for scanning Bitcoin transactions for steganography",
    version="1.0.0",
    lifespan=lifespan
)

# Health endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Service health check with scanner and Bitcoin node status"""
    scanner_status = {
        "model_loaded": scanner_instance is not None,
        "model_version": "v4-prod" if scanner_instance else "none",
        "model_path": "models/detector_balanced.onnx",
        "device": "cpu"  # Would detect actual device
    }
    
    bitcoin_status = {
        "node_connected": bitcoin_client.connected,
        "node_url": bitcoin_client.node_url,
        "block_height": await bitcoin_client.get_block_height()
    }
    
    if FASTAPI_AVAILABLE:
        return HealthResponse(
            status="healthy" if scanner_instance and bitcoin_client.connected else "degraded",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            scanner=scanner_status,
            bitcoin=bitcoin_status
        )
    else:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "scanner": scanner_status,
            "bitcoin": bitcoin_status
        }

@app.get("/info", response_model=InfoResponse, tags=["Info"])
async def api_info():
    """API information and capabilities"""
    if FASTAPI_AVAILABLE:
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
                "inscribe": "/inscribe"
            }
        )
    else:
        return {
            "name": "Starlight Bitcoin Steganography Scanner",
            "version": "1.0.0",
            "description": "AI-powered steganography detection for Bitcoin transaction images",
            "supported_formats": ["png", "jpg", "jpeg", "gif", "bmp", "webp"],
            "stego_methods": ["alpha", "palette", "lsb.rgb", "exif", "raw"],
            "max_image_size": 10485760,
            "endpoints": {
                "scan_tx": "/scan/transaction",
                "scan_image": "/scan/image", 
                "batch_scan": "/scan/batch",
                "extract": "/extract",
                "inscribe": "/inscribe"
            }
        }

# Transaction scanning endpoint
@app.post("/scan/transaction", response_model=TransactionScanResponse, tags=["Scanning"])
async def scan_transaction(
    request: TransactionScanRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
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
            extracted_images = await bitcoin_client.extract_images(request.transaction_id)
            
            for img_data in extracted_images:
                # Scan image asynchronously
                scan_result = await scan_image_async(
                    img_data["data"], 
                    request.scan_options, 
                    request_id
                )
                
                images.append(ImageScanResult(
                    index=img_data["index"],
                    size_bytes=img_data["size_bytes"],
                    format=img_data["format"],
                    scan_result=scan_result
                ))
        
        # Calculate summary
        stego_detected = any(img.scan_result.is_stego for img in images)
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        scan_summary = {
            "images_found": len(images),
            "images_scanned": len(images),
            "stego_detected": stego_detected,
            "processing_time_ms": processing_time
        }
        
        if FASTAPI_AVAILABLE:
            return TransactionScanResponse(
                transaction_id=request.transaction_id,
                block_height=tx_data["block_height"],
                timestamp=tx_data["timestamp"],
                scan_results=scan_summary,
                images=images,
                request_id=request_id
            )
        else:
            return {
                "transaction_id": request.transaction_id,
                "block_height": tx_data["block_height"],
                "timestamp": tx_data["timestamp"],
                "scan_results": scan_summary,
                "images": images,
                "request_id": request_id
            }
    
    except Exception as e:
        logger.error(f"Error scanning transaction {request.transaction_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Transaction scan failed: {str(e)}")

# Direct image scanning endpoint
@app.post("/scan/image", response_model=DirectImageScanResponse, tags=["Scanning"])
async def scan_image(
    image: UploadFile = File(...),
    extract_message: bool = Form(True),
    confidence_threshold: float = Form(0.5),
    include_metadata: bool = Form(True),
    api_key: str = Depends(verify_api_key)
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
            include_metadata=include_metadata
        )
        
        # Scan image
        scan_result = await scan_image_async(image_data, options, request_id)
        
        # Get image info
        image_info = {
            "filename": image.filename,
            "size_bytes": len(image_data),
            "format": image.filename.split('.')[-1].lower() if '.' in image.filename else 'unknown'
        }
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        if FASTAPI_AVAILABLE:
            return DirectImageScanResponse(
                scan_result=scan_result,
                image_info=image_info,
                processing_time_ms=processing_time,
                request_id=request_id
            )
        else:
            return {
                "scan_result": scan_result,
                "image_info": image_info,
                "processing_time_ms": processing_time,
                "request_id": request_id
            }
    
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
    api_key: str = Depends(verify_api_key)
):
    """Embed a text message into an image and save it for Stargate to broadcast."""
    request_id = str(uuid.uuid4())
    try:
        if not message:
            raise HTTPException(status_code=400, detail="Message is required for inscription")
        image_bytes = await image.read()
        if len(image_bytes) > 10485760:
            raise HTTPException(status_code=413, detail="Image too large")
        try:
            stego_bytes = embed_message_to_image(image_bytes, method, message)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except RuntimeError as re:
            raise HTTPException(status_code=503, detail=str(re))

        safe_name = os.path.basename(image.filename) or "inscribe.png"

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
                resp = requests.post(ingest_url, json=payload, headers=headers, timeout=10)
                ingest_result = {"status_code": resp.status_code, "response": resp.text}
            except Exception as e:
                ingest_result = {"error": str(e)}

        response_payload = {
            "request_id": request_id,
            "method": method,
            "message_length": len(message.encode("utf-8")),
            "output_file": safe_name,
            "image_bytes": len(stego_bytes),
            "status": "ingested" if ingest_result else "pending_upload",
            "note": "Ingested to Stargate via REST" if ingest_result else "No ingest URL configured",
            "ingest": ingest_result,
        }
        if FASTAPI_AVAILABLE:
            return InscribeResponse(**response_payload)
        return response_payload

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
    api_key: str = Depends(verify_api_key)
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
                    tx_data = await bitcoin_client.get_transaction(item.transaction_id)
                    images = await bitcoin_client.extract_images(item.transaction_id)
                    
                    item_stego_count = 0
                    for img_data in images:
                        scan_result = await scan_image_async(
                            img_data["data"],
                            request.scan_options,
                            request_id
                        )
                        if scan_result.is_stego:
                            item_stego_count += 1
                    
                    if item_stego_count > 0:
                        stego_count += 1
                    
                    results.append(BatchItemResult(
                        item_id=item.transaction_id,
                        type="transaction",
                        status="completed",
                        stego_detected=item_stego_count > 0,
                        images_with_stego=item_stego_count,
                        total_images=len(images)
                    ))
                
                elif item.type == "image":
                    # Scan base64 image
                    image_data = base64.b64decode(item.image_data)
                    scan_result = await scan_image_async(
                        image_data,
                        request.scan_options,
                        request_id
                    )
                    
                    if scan_result.is_stego:
                        stego_count += 1
                    
                    results.append(BatchItemResult(
                        item_id=f"image_{len(results)}",
                        type="image",
                        status="completed",
                        stego_detected=scan_result.is_stego,
                        images_with_stego=1 if scan_result.is_stego else 0,
                        total_images=1
                    ))
            
            except Exception as e:
                results.append(BatchItemResult(
                    item_id=item.transaction_id or f"image_{len(results)}",
                    type=item.type,
                    status="failed",
                    stego_detected=False,
                    images_with_stego=0,
                    total_images=0,
                    error=str(e)
                ))
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        if FASTAPI_AVAILABLE:
            return BatchScanResponse(
                batch_id=batch_id,
                total_items=len(request.items),
                processed_items=len([r for r in results if r.status == "completed"]),
                stego_detected=stego_count,
                processing_time_ms=processing_time,
                results=results,
                request_id=request_id
            )
        else:
            return {
                "batch_id": batch_id,
                "total_items": len(request.items),
                "processed_items": len([r for r in results if r.status == "completed"]),
                "stego_detected": stego_count,
                "processing_time_ms": processing_time,
                "results": results,
                "request_id": request_id
            }
    
    except Exception as e:
        logger.error(f"Error in batch scan: {e}")
        raise HTTPException(status_code=500, detail=f"Batch scan failed: {str(e)}")

# Message extraction endpoint
@app.post("/extract", response_model=ExtractResponse, tags=["Extraction"])
async def extract_message(
    image: UploadFile = File(...),
    method: Optional[str] = Form(None),
    force_extract: bool = Form(False),
    api_key: str = Depends(verify_api_key)
):
    """Extract hidden messages from a steganographic image"""
    start_time = datetime.utcnow()
    request_id = str(uuid.uuid4())
    
    try:
        # Read image data
        image_data = await image.read()
        
        # Save to temporary file for extraction
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
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
                        "bits_extracted": len(result.get("extracted_message", "")) * 8 if result.get("extracted_message") else 0,
                        "encoding": "utf-8",
                        "corruption_detected": False
                    }
                )
            else:
                # Stub response
                extraction_result = ExtractionResult(
                    message_found=False,
                    extraction_details={
                        "bits_extracted": 0,
                        "encoding": "utf-8",
                        "corruption_detected": False
                    }
                )
            
            image_info = {
                "filename": image.filename,
                "size_bytes": len(image_data),
                "format": image.filename.split('.')[-1].lower() if '.' in image.filename else 'unknown'
            }
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if FASTAPI_AVAILABLE:
                return ExtractResponse(
                    extraction_result=extraction_result,
                    image_info=image_info,
                    processing_time_ms=processing_time,
                    request_id=request_id
                )
            else:
                return {
                    "extraction_result": extraction_result,
                    "image_info": image_info,
                    "processing_time_ms": processing_time,
                    "request_id": request_id
                }
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    except Exception as e:
        logger.error(f"Error extracting message: {e}")
        raise HTTPException(status_code=500, detail=f"Message extraction failed: {str(e)}")

# Transaction lookup endpoint
@app.get("/transaction/{txid}", tags=["Transaction"])
async def get_transaction(
    txid: str,
    include_images: bool = Query(False),
    image_format: str = Query("info"),
    api_key: str = Depends(verify_api_key)
):
    """Get transaction details and available images"""
    try:
        # Validate transaction ID
        if not len(txid) == 64 or not all(c in '0123456789abcdefABCDEF' for c in txid):
            raise HTTPException(status_code=400, detail="Invalid transaction ID format")
        
        # Get transaction data
        tx_data = await bitcoin_client.get_transaction(txid)
        
        response_data = {
            "transaction_id": txid,
            "block_height": tx_data["block_height"],
            "timestamp": tx_data["timestamp"],
            "status": tx_data["status"],
            "total_images": 0
        }
        
        if include_images:
            images = await bitcoin_client.extract_images(txid)
            response_data["images"] = []
            
            for img in images:
                img_data = {
                    "index": img["index"],
                    "size_bytes": img["size_bytes"],
                    "format": img["format"]
                }
                
                if image_format == "base64":
                    img_data["data"] = base64.b64encode(img["data"]).decode('utf-8')
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
        raise HTTPException(status_code=500, detail=f"Transaction lookup failed: {str(e)}")

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
                    "request_id": str(uuid.uuid4())
                }
            }
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
                    "request_id": str(uuid.uuid4())
                }
            }
        )

# Block scanning endpoint
@app.post("/scan/block", response_model=BlockScanResponse, tags=["Scanning"])
async def scan_block(
    request: BlockScanRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
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
            raise HTTPException(status_code=404, detail=f"Block {request.block_height} not found")
        block_dir = matches[0]
        
        # Load block data
        block_json_path = os.path.join(block_dir, "block.json")
        block_data = {}
        if os.path.exists(block_json_path):
            with open(block_json_path, 'r') as f:
                block_data = json.load(f)
        
        # Load existing inscriptions if available
        inscriptions_json_path = os.path.join(block_dir, "inscriptions.json")
        existing_inscriptions = {}
        if os.path.exists(inscriptions_json_path):
            with open(inscriptions_json_path, 'r') as f:
                existing_inscriptions = json.load(f)
        
        # Scan images in block
        images_dir = os.path.join(block_dir, "images")
        scanned_inscriptions = []
        images_scanned = 0
        stego_detected = 0
        
        if os.path.exists(images_dir):
            image_files = [f for f in os.listdir(images_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'))]
            
            for image_file in image_files:
                image_path = os.path.join(images_dir, image_file)
                try:
                    # Scan image
                    scan_result = await scan_image_async(
                        open(image_path, 'rb').read(),
                        request.scan_options,
                        request_id
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
                            "file_path": f"images/{image_file}"
                        }
                    
                    # Add scan result to inscription
                    if FASTAPI_AVAILABLE and hasattr(scan_result, 'dict'):
                        inscription_info["scan_result"] = scan_result.dict()
                        if getattr(scan_result, "is_stego", False):
                            stego_detected += 1
                    else:
                        inscription_info["scan_result"] = scan_result
                        if isinstance(scan_result, dict) and scan_result.get("is_stego"):
                            stego_detected += 1
                    
                    scanned_inscriptions.append(inscription_info)
                    images_scanned += 1
                    
                except Exception as e:
                    logger.error(f"Error scanning image {image_file}: {e}")
                    # Add inscription without scan result
                    scanned_inscriptions.append({
                        "tx_id": f"unknown_{image_file}",
                        "input_index": 0,
                        "content_type": f"image/{image_file.split('.')[-1]}",
                        "content": f"Image file: {image_file}",
                        "size_bytes": os.path.getsize(image_path),
                        "file_name": image_file,
                        "file_path": f"images/{image_file}",
                        "scan_result": None,
                        "error": str(e)
                    })
                    images_scanned += 1
        
        # Update block data with scan results
        updated_block_data = block_data.copy()
        updated_block_data["stego_detected"] = stego_detected
        updated_block_data["images_scanned"] = images_scanned
        updated_block_data["scan_timestamp"] = int(start_time.timestamp())
        
        # Save updated block data
        with open(block_json_path, 'w') as f:
            json.dump(updated_block_data, f, indent=2)
        
        # Update inscriptions with scan results
        updated_inscriptions = existing_inscriptions.copy()
        updated_inscriptions["inscriptions"] = scanned_inscriptions
        
        with open(inscriptions_json_path, 'w') as f:
            json.dump(updated_inscriptions, f, indent=2)
        
        # Update global inscriptions JSON
        global_inscriptions_path = os.path.join(blocks_dir, "global_inscriptions.json")
        global_inscriptions = {}
        
        if os.path.exists(global_inscriptions_path):
            with open(global_inscriptions_path, 'r') as f:
                global_inscriptions = json.load(f)
        
        block_key = f"{request.block_height}_{block_data.get('hash', 'unknown')}"
        global_inscriptions[block_key] = {
            "hash": block_data.get("hash", "unknown"),
            "height": request.block_height,
            "image_count": images_scanned,
            "inscriptions": scanned_inscriptions,
            "timestamp": block_data.get("timestamp", 0),
            "stego_detected": stego_detected,
            "scan_timestamp": int(start_time.timestamp())
        }
        
        with open(global_inscriptions_path, 'w') as f:
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
        
        if FASTAPI_AVAILABLE:
            return BlockScanResponse(
                block_height=request.block_height,
                block_hash=block_hash or "unknown",
                timestamp=block_timestamp,
                total_inscriptions=len(scanned_inscriptions),
                images_scanned=images_scanned,
                stego_detected=stego_detected,
                processing_time_ms=processing_time,
                inscriptions=scanned_inscriptions,
                request_id=request_id
            )
        else:
            return {
                "block_height": request.block_height,
                "block_hash": block_hash or "unknown",
                "timestamp": block_timestamp,
                "total_inscriptions": len(scanned_inscriptions),
                "images_scanned": images_scanned,
                "stego_detected": stego_detected,
                "processing_time_ms": processing_time,
                "inscriptions": scanned_inscriptions,
                "request_id": request_id
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scanning block {request.block_height}: {e}")
        raise HTTPException(status_code=500, detail=f"Block scan failed: {str(e)}")

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API root endpoint with basic information"""
    return {
        "name": "Starlight Bitcoin Scanning API",
        "version": "1.0.0",
        "description": "API for scanning Bitcoin transactions for steganography",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "scan_transaction": "/scan/transaction",
            "scan_image": "/scan/image",
            "batch_scan": "/scan/batch",
            "scan_block": "/scan/block",
            "extract": "/extract",
            "inscribe": "/inscribe",
            "get_transaction": "/transaction/{txid}"
        },
        "documentation": "/docs",
        "timestamp": datetime.utcnow().isoformat(),
        "fastapi_available": FASTAPI_AVAILABLE,
        "scanner_available": SCANNER_AVAILABLE
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
        logger.info("To run the full API, install: pip install fastapi uvicorn pydantic")
        
        # Test the stub functionality
        print("\n=== Starlight Bitcoin Scanning API Stub Test ===")
        
        async def test_stub():
            health = await health_check()
            print(f"Health status: {health['status']}")
            print(f"Scanner loaded: {health['scanner']['model_loaded']}")
            print(f"Bitcoin node connected: {health['bitcoin']['node_connected']}")
            
            print("\nTesting transaction scan...")
            try:
                tx_request = TransactionScanRequest(
                    transaction_id="f4184fc596403b9d638783cf57adfe4c75c605f6356fbc91338530e9831e9e16"
                )
                result = await scan_transaction(tx_request, BackgroundTasks(), "demo-key")
                print(f"Transaction scanned: {result['transaction_id']}")
                print(f"Images found: {result['scan_results']['images_found']}")
                print(f"Stego detected: {result['scan_results']['stego_detected']}")
            except Exception as e:
                print(f"Transaction scan test failed: {e}")
        
        asyncio.run(test_stub())
        
        print("\n=== Stub test completed ===")
        print("Install FastAPI dependencies for full functionality")
