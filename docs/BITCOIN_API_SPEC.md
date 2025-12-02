# Starlight Bitcoin Steganography Scanning API Specification

**Version**: 1.1  
**Format**: OpenAPI 3.0  
**Updated**: December 1, 2025  

---

## 🎯 Overview

The Starlight Bitcoin Scanning API provides real-time steganography detection for images embedded in Bitcoin transactions. This API integrates with the existing Starlight scanner to analyze images extracted from blockchain transactions and detect hidden data using various steganographic techniques.

### Base URL
```
Production: https://api.starlight.ai/bitcoin/v1
Development: http://localhost:8080/bitcoin/v1
```

### Authentication
- **Bearer Token**: JWT-based authentication for external AI services
- **Service-to-Service**: mTLS for internal service communication
- **Public Endpoints**: `/health` and `/info` (read-only)

---

## 📊 API Endpoints

### Health & Information

#### GET /health
Service health check with scanner status.

**SLA**: <100ms response time

**Responses**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-26T10:00:00Z",
  "version": "v1.0.0",
  "scanner": {
    "model_loaded": true,
    "model_version": "v4-prod",
    "model_path": "models/detector_balanced.onnx",
    "device": "cpu"
  },
  "bitcoin": {
    "node_connected": true,
    "node_url": "https://blockstream.info/api",
    "block_height": 856789
  }
}
```

#### GET /info
API information and capabilities.

**Response**:
```json
{
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
    "scan_block": "/scan/block",
    "extract": "/extract",
    "inscribe": "/inscribe"
  }
}
```

---

### Transaction Scanning

#### POST /scan/transaction
Scan a Bitcoin transaction for steganography in embedded images.

**SLA**: <5s response time

**Request Body**:
```json
{
  "transaction_id": "f4184fc596403b9d638783cf57adfe4c75c605f6356fbc91338530e9831e9e16",
  "extract_images": true,
  "scan_options": {
    "extract_message": true,
    "confidence_threshold": 0.5,
    "include_metadata": true
  }
}
```

**Response**:
```json
{
  "transaction_id": "f4184fc596403b9d638783cf57adfe4c75c605f6356fbc91338530e9831e9e16",
  "block_height": 170000,
  "timestamp": "2025-11-26T10:00:00Z",
  "scan_results": {
    "images_found": 2,
    "images_scanned": 2,
    "stego_detected": 1,
    "processing_time_ms": 1250
  },
  "images": [
    {
      "index": 0,
      "size_bytes": 1048576,
      "format": "png",
      "scan_result": {
        "is_stego": false,
        "stego_probability": 0.02,
        "confidence": 0.98,
        "prediction": "clean"
      }
    },
    {
      "index": 1,
      "size_bytes": 2097152,
      "format": "png",
      "scan_result": {
        "is_stego": true,
        "stego_probability": 0.87,
        "stego_type": "alpha",
        "confidence": 0.87,
        "prediction": "stego",
        "extracted_message": "Hello from the blockchain!",
        "method_id": 0
      }
    }
  ],
  "request_id": "req_123456789"
}
```

**Status Codes**:
- `200 OK` - Transaction scanned successfully
- `400 Bad Request` - Invalid transaction ID or parameters
- `404 Not Found` - Transaction not found
- `422 Unprocessable Entity` - No images found in transaction
- `500 Internal Server Error` - Scanning service error

---

### Direct Image Scanning

#### POST /scan/image
Scan a directly uploaded image for steganography.

**SLA**: <2s response time

**Request**: `multipart/form-data`
- `image`: Image file (required)
- `extract_message`: Boolean (optional, default: true)
- `confidence_threshold`: Float (optional, default: 0.5)
- `include_metadata`: Boolean (optional, default: true)

**Response**:
```json
{
  "scan_result": {
    "is_stego": true,
    "stego_probability": 0.92,
    "stego_type": "lsb.rgb",
    "confidence": 0.92,
    "prediction": "stego",
    "method_id": 2,
    "extracted_message": "Secret data hidden in LSB"
  },
  "image_info": {
    "filename": "example.png",
    "size_bytes": 1048576,
    "format": "png",
    "dimensions": "256x256"
  },
  "processing_time_ms": 450,
  "request_id": "req_987654321"
}
```

---

### Batch Scanning

#### POST /scan/batch
Scan multiple transactions or images in a batch.

**SLA**: <10s response time (up to 50 items)

**Request Body**:
```json
{
  "items": [
    {
      "type": "transaction",
      "transaction_id": "f4184fc596403b9d638783cf57adfe4c75c605f6356fbc91338530e9831e9e16"
    },
    {
      "type": "transaction",
      "transaction_id": "a1075db55d416d3ca199f55b6084e2115b9345e16c5cf302fc80e9d5fbf5d48d"
    }
  ],
  "scan_options": {
    "extract_message": true,
    "confidence_threshold": 0.5,
    "max_concurrent": 5
  }
}
```

**Response**:
```json
{
  "batch_id": "batch_abc123",
  "total_items": 2,
  "processed_items": 2,
  "stego_detected": 1,
  "processing_time_ms": 3200,
  "results": [
    {
      "item_id": "f4184fc596403b9d638783cf57adfe4c75c605f6356fbc91338530e9831e9e16",
      "type": "transaction",
      "status": "completed",
      "stego_detected": true,
      "images_with_stego": 1,
      "total_images": 2
    },
    {
      "item_id": "a1075db55d416d3ca199f55b6084e2115b9345e16c5cf302fc80e9d5fbf5d48d",
      "type": "transaction",
      "status": "completed",
      "stego_detected": false,
      "images_with_stego": 0,
      "total_images": 1
    }
  ],
  "request_id": "req_batch_456"
}
```

---

### Block Scanning

#### POST /scan/block
Scan a locally-synced block folder for inscription images and return per-image stego results (used by Stargate for offline batch processing).

**SLA**: Depends on block size; synchronous call—queue/worker recommended for production.

**Request Body**:
```json
{
  "block_height": 815000,
  "scan_options": {
    "extract_message": true,
    "confidence_threshold": 0.5,
    "include_metadata": true
  }
}
```

**Response**:
```json
{
  "block_height": 815000,
  "block_hash": "0000000000000000000123456789abcdef",
  "timestamp": 1701427200,
  "total_inscriptions": 32,
  "images_scanned": 32,
  "stego_detected": 1,
  "processing_time_ms": 4100,
  "inscriptions": [
    {
      "tx_id": "abc123...",
      "input_index": 0,
      "content_type": "image/png",
      "content": "Image file: insc-0001.png",
      "size_bytes": 20480,
      "file_name": "insc-0001.png",
      "file_path": "images/insc-0001.png",
      "scan_result": {
        "is_stego": true,
        "stego_probability": 0.92,
        "stego_type": "alpha",
        "confidence": 0.94,
        "prediction": "stego"
      }
    }
  ],
  "request_id": "7c1f8b4c-1fd1-4e2d-9f1f-2a6b1d35b521"
}
```

Notes:
- Expects block data on disk (e.g., `blocks/{height}_*/images/*.png` with `block.json`).
- `stego_detected` increments when any inscription `scan_result.is_stego` is true.
- Prefer asynchronous job submission for large blocks.

---

### Message Extraction

#### POST /extract
Extract hidden messages from a steganographic image.

**SLA**: <3s response time

**Request**: `multipart/form-data`
- `image`: Image file (required)
- `method`: String (optional, auto-detect if not provided)
- `force_extract`: Boolean (optional, default: false)

**Response**:
```json
{
  "extraction_result": {
    "message_found": true,
    "message": "This is a hidden message in the image",
    "method_used": "alpha",
    "method_confidence": 0.87,
    "extraction_details": {
      "bits_extracted": 256,
      "encoding": "utf-8",
      "corruption_detected": false
    }
  },
  "image_info": {
    "filename": "stego_image.png",
    "size_bytes": 1048576,
    "format": "png"
  },
  "processing_time_ms": 280,
  "request_id": "req_extract_789"
}
```

---

### Inscription Preparation

#### POST /inscribe
Embed a text payload into an image and save it to disk for Stargate to broadcast via OP_RETURN/Ordinals (no on-chain write here).

**SLA**: <2s

**Request**: `multipart/form-data`
- `image` (file, required) – cover image (<=10MB)
- `message` (string, required) – UTF-8 text payload to embed
- `method` (string, optional, default `alpha`) – one of `alpha|lsb|palette|exif|eoi`

**Response**:
```json
{
  "request_id": "c5c2cf9a-27c3-4c28-b2d8-9bba2b4a6a11",
  "method": "alpha",
  "message_length": 42,
  "output_file": "inscriptions/pending/20241201T120000Z_ab12cd34_cover.png",
  "image_bytes": 58213,
  "status": "pending_upload",
  "note": "Ready for Stargate uploader to broadcast via OP_RETURN or Ordinals"
}
```

Notes:
- Output directory defaults to `inscriptions/pending` (override via `STARLIGHT_INSCRIPTION_OUTBOX`).
- Stargate should poll the outbox and perform the on-chain inscription.
- Auth: Bearer token required (except /health and /info).

---

### Transaction Lookup

#### GET /transaction/{txid}
Get transaction details and available images.

**Query Parameters**:
- `include_images`: Boolean (optional, default: false)
- `image_format`: String (optional, default: "info")

**Response**:
```json
{
  "transaction_id": "f4184fc596403b9d638783cf57adfe4c75c605f6356fbc91338530e9831e9e16",
  "block_height": 170000,
  "timestamp": "2025-11-26T10:00:00Z",
  "status": "confirmed",
  "images": [
    {
      "index": 0,
      "size_bytes": 1048576,
      "format": "png",
      "data_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ],
  "total_images": 1
}
```

---

## 🔒 Rate Limiting

| Endpoint | Rate Limit | Burst | Time Window |
|----------|------------|-------|-------------|
| POST /scan/transaction | 60 req/min | 10 | 1 minute |
| POST /scan/image | 120 req/min | 20 | 1 minute |
| POST /scan/batch | 10 req/min | 5 | 1 minute |
| POST /extract | 30 req/min | 10 | 1 minute |
| GET /transaction/{txid} | 100 req/min | 20 | 1 minute |

---

## 📝 Data Models

### ScanResult
```json
{
  "is_stego": "boolean",
  "stego_probability": "number (0-1)",
  "stego_type": "string (alpha|palette|lsb.rgb|exif|raw)",
  "confidence": "number (0-1)",
  "prediction": "string (stego|clean)",
  "method_id": "integer (0-4)",
  "extracted_message": "string (optional)",
  "extraction_error": "string (optional)"
}
```

### TransactionScanRequest
```json
{
  "transaction_id": "string (64-char hex)",
  "extract_images": "boolean (default: true)",
  "scan_options": {
    "extract_message": "boolean (default: true)",
    "confidence_threshold": "number (0-1, default: 0.5)",
    "include_metadata": "boolean (default: true)"
  }
}
```

### BatchScanRequest
```json
{
  "items": [
    {
      "type": "string (transaction|image)",
      "transaction_id": "string (for transaction type)",
      "image_data": "string (base64, for image type)"
    }
  ],
  "scan_options": {
    "extract_message": "boolean",
    "confidence_threshold": "number",
    "max_concurrent": "integer (1-10)"
  }
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
| `INVALID_TX_ID` | 400 | Invalid Bitcoin transaction ID |
| `TX_NOT_FOUND` | 404 | Transaction not found on blockchain |
| `NO_IMAGES_FOUND` | 422 | No images found in transaction |
| `IMAGE_TOO_LARGE` | 413 | Image exceeds size limit |
| `UNSUPPORTED_FORMAT` | 415 | Image format not supported |
| `SCAN_FAILED` | 500 | Steganography scan failed |
| `EXTRACTION_FAILED` | 500 | Message extraction failed |
| `BITCOIN_NODE_ERROR` | 503 | Bitcoin node unavailable |

---

## 🛰️ Stargate Integration Notes (Production Readiness)
- Prefer async workers for `/scan/block`; store results in Postgres JSONB for flexible querying. Example schema: `CREATE TABLE block_scans (block_height INT PRIMARY KEY, block_hash TEXT, scanned_at TIMESTAMPTZ, payload JSONB, stego_detected INT, images_scanned INT);`
- Index JSONB for hot fields: `CREATE INDEX ON block_scans USING GIN (payload jsonb_path_ops);` plus btree on `(block_height)` and `(scanned_at)`.
- Upstream extraction logic should handle dedupe/retries per `block_hash`; treat `stego_detected` as a derived column from the payload.
- Inscription flow: Stargate watches `inscriptions/pending`, uploads the file, records resulting txid back into its own table keyed by `output_file`.

---

## 🔗 Integration Examples

### Python Client
```python
import requests
import json
import base64

class StarlightBitcoinClient:
    def __init__(self, base_url, api_key=None):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'StarlightBitcoinClient/1.0'
        }
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
    
    def scan_transaction(self, tx_id, extract_images=True, options=None):
        """Scan a Bitcoin transaction for steganography"""
        payload = {
            "transaction_id": tx_id,
            "extract_images": extract_images,
            "scan_options": options or {}
        }
        
        response = requests.post(
            f'{self.base_url}/scan/transaction',
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def scan_image(self, image_path, options=None):
        """Scan a local image file for steganography"""
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {}
            if options:
                data.update(options)
            
            response = requests.post(
                f'{self.base_url}/scan/image',
                headers={'Authorization': self.headers['Authorization']},
                files=files,
                data=data
            )
        response.raise_for_status()
        return response.json()
    
    def batch_scan(self, items, options=None):
        """Batch scan multiple transactions"""
        payload = {
            "items": items,
            "scan_options": options or {}
        }
        
        response = requests.post(
            f'{self.base_url}/scan/batch',
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()

# Usage example
client = StarlightBitcoinClient('https://api.starlight.ai/bitcoin/v1', 'your-api-key')

# Scan a transaction
result = client.scan_transaction(
    'f4184fc596403b9d638783cf57adfe4c75c605f6356fbc91338530e9831e9e16'
)

if result['scan_results']['stego_detected']:
    print(f"Steganography detected in transaction {result['transaction_id']}")
    for img in result['images']:
        if img['scan_result']['is_stego']:
            print(f"  Image {img['index']}: {img['scan_result']['stego_type']}")
            if 'extracted_message' in img['scan_result']:
                print(f"  Message: {img['scan_result']['extracted_message']}")

# Batch scan
batch_items = [
    {"type": "transaction", "transaction_id": "tx1..."},
    {"type": "transaction", "transaction_id": "tx2..."}
]
batch_result = client.batch_scan(batch_items)
print(f"Batch scan complete: {batch_result['stego_detected']} items with stego")
```

### cURL Examples
```bash
# Health check
curl -X GET "https://api.starlight.ai/bitcoin/v1/health"

# Scan transaction
curl -X POST "https://api.starlight.ai/bitcoin/v1/scan/transaction" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "transaction_id": "f4184fc596403b9d638783cf57adfe4c75c605f6356fbc91338530e9831e9e16",
    "extract_images": true,
    "scan_options": {
      "extract_message": true,
      "confidence_threshold": 0.5
    }
  }'

# Scan image file
curl -X POST "https://api.starlight.ai/bitcoin/v1/scan/image" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "image=@example.png" \
  -F "extract_message=true" \
  -F "confidence_threshold=0.7"

# Batch scan
curl -X POST "https://api.starlight.ai/bitcoin/v1/scan/batch" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "items": [
      {"type": "transaction", "transaction_id": "tx1..."},
      {"type": "transaction", "transaction_id": "tx2..."}
    ]
  }'
```

---

## 🔧 Technical Implementation

### Bitcoin Integration
- **Node Connection**: Uses Blockstream API or self-hosted Bitcoin Core node
- **OP_RETURN Extraction**: Parses OP_RETURN outputs for embedded data
- **Multi-format Support**: Handles images embedded in various transaction formats
- **Block Height Tracking**: Maintains real-time blockchain synchronization

### Scanner Integration
- **Model Loading**: Leverages existing `scanner.py` StarlightScanner class
- **Parallel Processing**: Supports concurrent image scanning
- **Memory Management**: Efficient handling of large batch operations
- **Error Recovery**: Graceful degradation for partial failures

### Performance Optimizations
- **Image Caching**: Caches extracted images from frequently accessed transactions
- **Batch Processing**: Optimized for high-throughput batch operations
- **Async Processing**: Non-blocking I/O for Bitcoin node queries
- **Resource Pooling**: Reuses model instances across requests

---

## 📊 Monitoring & Observability

### API Metrics
The Bitcoin Scanning API exposes metrics compatible with the existing monitoring infrastructure:

```
# Bitcoin scanning metrics
starlight_bitcoin_scans_total{method="transaction",status="success"} 1500
starlight_bitcoin_scan_duration_seconds{method="image",quantile="0.95"} 0.45
starlight_bitcoin_stego_detection_rate{method="transaction"} 0.0199

# Blockchain metrics
starlight_bitcoin_node_requests_total{endpoint="tx"} 5000
starlight_bitcoin_block_height 856789
starlight_bitcoin_images_extracted_total 2500
```

### Health Check Endpoints
- `/health` - Overall service health
- `/health/bitcoin` - Bitcoin node connectivity
- `/health/scanner` - Scanner model status
- `/metrics` - Prometheus metrics endpoint

---

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-26 | Initial release with core scanning endpoints |
| 0.9 | 2025-11-25 | Beta testing with Bitcoin Core integration |
| 0.8 | 2025-11-20 | Added batch scanning capabilities |
| 0.7 | 2025-11-15 | Integrated with Starlight scanner v4 |

---

## 📞 Support & Contact

### API Support
- **Documentation**: https://docs.starlight.ai/bitcoin-api
- **Issues**: bitcoin-api@starlight.ai
- **Status Page**: https://status.starlight.ai

### SLA Information
- **Availability**: 99.9% (monthly)
- **Support Response**: <1 hour for P1 incidents
- **Maintenance Window**: Sundays 02:00-04:00 UTC

---

**Specification Version**: 1.0  
**Last Updated**: November 26, 2025  
**Next Review**: December 26, 2025  
**Maintainer: AI Software Architect (Bitcoin Integration)**
