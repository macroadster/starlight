# Starlight Dynamic API

Secure dynamic loading of agent-developed functions into FastAPI endpoints.

## Overview

The Dynamic API system allows agents to develop Python code in isolated sandboxes and have those functions dynamically loaded as FastAPI endpoints. This enables:

- **Secure Function Loading**: Functions are loaded from isolated directories based on `visible_pixel_hash`
- **API Key Authentication**: Write operations require `STARGATE_API_KEY` authentication
- **Security Validation**: AST parsing prevents dangerous operations
- **Runtime Endpoint Registration**: Functions become accessible as HTTP endpoints
- **Isolation**: Each contract's code runs in its own directory

## Architecture

```
Agent develops code → Uploads to sandbox → Dynamic API loads → FastAPI endpoint available
     ↓                        ↓                 ↓                   ↓
UPLOADS_DIR/results/[hash]/   AST validation   Function registration  HTTP API access
├── clean/                   ← Security ←      Isolated execution ← API key auth
├── stego/                   validation        → Endpoint mapping    → Public access
└── agent_code.py           (no dangerous ops)                    (read-only)
```

## Security Features

### 1. **API Key Authentication**
- **Write operations** (load/unload) require valid `STARGATE_API_KEY`
- **Read operations** (list/call) are publicly available for loaded functions
- Uses Bearer token authentication

### 2. **Sandbox Isolation**
- Functions loaded from `UPLOADS_DIR/results/[visible_pixel_hash]/`
- Each contract has its own isolated directory
- Working directory changed before function execution

### 3. **AST Security Validation**
- Blocks dangerous imports: `os`, `subprocess`, `socket`, `requests`, `urllib`
- Blocks dangerous functions: `open`, `exec`, `eval`, `compile`
- Syntax validation before loading

### 4. **Memory Isolation**
- Each function loaded as separate module
- Module tracking for cleanup
- Function execution isolation

## API Endpoints

### Write Operations (Require API Key)

#### Load Function
```http
POST /load-function
Authorization: Bearer {STARGATE_API_KEY}
Content-Type: application/json

{
  "visible_pixel_hash": "abcdef1234567890abcdef1234567890abcdef12",
  "function_name": "hello_world",
  "module_name": "agent_functions",
  "endpoint_path": "/agent/hello",
  "method": "GET"
}
```

#### Unload Function
```http
DELETE /unload-function/{visible_pixel_hash}/{function_name}
Authorization: Bearer {STARGATE_API_KEY}
```

### Read Operations (No Authentication Required)

#### List Functions
```http
GET /list-functions
```

#### Call Function Directly
```http
GET /function/{visible_pixel_hash}/{function_name}
```

#### Call via Dynamic Endpoint
```http
GET /agent/{visible_pixel_hash}/{function_name}
POST /agent/{visible_pixel_hash}/{function_name}
```

#### Health Check
```http
GET /health
```

## Usage Examples

### 1. Start the Server
```bash
# Set environment variables
export STARGATE_API_KEY="your-secret-api-key"
export UPLOADS_DIR="/data/uploads"

# Start server
python3 dynamic_api_server.py
```

### 2. Agent Creates Function
Agent creates Python code in sandbox directory:
```python
# /data/uploads/results/[hash]/agent_functions.py
def calculate_fibonacci(n: int):
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
```

### 3. Load Function via API
```bash
curl -X POST "http://localhost:8000/load-function" \
  -H "Authorization: Bearer your-secret-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "visible_pixel_hash": "abcdef1234567890abcdef1234567890abcdef12",
    "function_name": "calculate_fibonacci",
    "module_name": "agent_functions",
    "endpoint_path": "/math/fibonacci"
  }'
```

### 4. Call the Function
```bash
# Call via dynamic endpoint
curl "http://localhost:8000/math/fibonacci?n=10"

# Call directly
curl "http://localhost:8000/function/abcdef1234567890abcdef1234567890abcdef12/calculate_fibonacci?n=10"
```

## Testing

Run the test suite:
```bash
# Start server first
python3 dynamic_api_server.py

# In another terminal:
python3 test_dynamic_api.py
```

The test suite:
- Creates a test sandbox with sample functions
- Tests all API endpoints
- Verifies security (unauthorized access)
- Tests function loading and execution

## Integration with Existing Apps

To integrate with existing FastAPI applications:

```python
from starlight.agents.dynamic_api import create_dynamic_app

# Create your existing app
app = FastAPI()

# Add dynamic loading capabilities
dynamic_app = create_dynamic_app()

# Mount or include routes as needed
```

## Configuration

Required environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `STARGATE_API_KEY` | Required | API key for write operations |
| `UPLOADS_DIR` | `/data/uploads` | Base directory for sandboxes |
| `DYNAMIC_API_HOST` | `0.0.0.0` | Server host |
| `DYNAMIC_API_PORT` | `8000` | Server port |

## Security Considerations

### Allowed vs Blocked Operations

**Allowed:**
- Basic Python functions
- Mathematical operations
- String manipulation
- Data structure operations
- Return JSON-serializable data

**Blocked:**
- File system access (`open`, file operations)
- Network access (`requests`, `socket`, `urllib`)
- Code execution (`exec`, `eval`, `compile`)
- Subprocess execution (`subprocess`, `os.system`)
- Dangerous imports

### Best Practices

1. **Always use HTTPS** in production
2. **Rotate API keys** regularly
3. **Monitor loaded functions** via `/list-functions`
4. **Validate function inputs** before execution
5. **Set resource limits** for function execution
6. **Audit sandbox directories** regularly

## Troubleshooting

### Common Issues

1. **403 Unauthorized**: Check `STARGATE_API_KEY`
2. **404 Not Found**: Verify `visible_pixel_hash` and sandbox directory
3. **403 Forbidden**: Check AST validation for blocked operations
4. **500 Internal Error**: Check server logs for detailed errors

### Debug Mode
```bash
# Enable debug logging
export ENVIRONMENT="development"
python3 dynamic_api_server.py
```

## File Structure

```
starlight/agents/
├── dynamic_loader.py    # Core loading logic
├── dynamic_api.py       # FastAPI endpoints
└── config.py           # Configuration

Root/
├── dynamic_api_server.py  # Standalone server
└── test_dynamic_api.py     # Test suite
```

## License

This system is part of the Starlight project for steganography detection AI training.