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
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Global instances
bitcoin_client = BitcoinNodeClient()
scanner_instance = None

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

# Dependency for API key authentication
async def verify_api_key(authorization: str = Header(None)):
    """Verify API key for protected endpoints"""
    if not FASTAPI_AVAILABLE:
        return "demo-api-key"
    
    if authorization is None:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    api_key = authorization.split(" ")[1]
    # Simple validation (replace with proper JWT verification in production)
    if api_key != "demo-api-key":
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
                "extract": "/extract"
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
                "extract": "/extract"
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
            "extract": "/extract",
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