# Starlight Sandbox Integration

The Bitcoin API now includes **sandbox capabilities** that allow Starlight to autonomously enhance its own functionality by loading agent-developed functions at runtime.

## 🚀 **Sandbox Architecture**

```
Agent Development → Dynamic Loading → Runtime Enhancement
       ↓                  ↓                    ↓
  Sandbox Code → Security Validation → API Extension
       ↓                  ↓                    ↓  
  Isolation via → AST Security → New Endpoints
visible_pixel_hash   Validation    Available Immediately
```

## 🔧 **How Self-Improvement Works**

### 1. **Agent Development**
- Agents develop Python functions in isolated sandboxes
- Functions are stored in `UPLOADS_DIR/results/[visible_pixel_hash]/`
- Each contract's code is isolated by its `visible_pixel_hash`

### 2. **Security Validation**
- AST parsing prevents dangerous operations
- Blocks: `os`, `subprocess`, `socket`, `requests`, `urllib`
- Blocks: `open`, `exec`, `eval`, `compile`
- Syntax validation before loading

### 3. **Dynamic Loading**
- Functions loaded via authenticated API endpoints
- Automatic FastAPI endpoint registration
- Real-time OpenAPI schema updates
- Memory-managed module tracking

## 📡 **New Self-Improvement Endpoints**

### Load Agent Function
```http
POST /self-improve/load
Authorization: Bearer {STARGATE_API_KEY}

{
  "visible_pixel_hash": "abcdef1234567890...",
  "function_name": "enhanced_scan",
  "module_name": "improvements",
  "improvement_type": "scan"
}
```

### Self-Improvement Status
```http
GET /self-improve/status
```
Returns:
- Total loaded functions
- Categorized improvements (scan/agent/api)
- Individual function details
- System capabilities

### Unload Function
```http
DELETE /self-improve/unload/{visible_pixel_hash}/{function_name}
Authorization: Bearer {STARGATE_API_KEY}
```

### Auto-Enhancement Trigger
```http
POST /self-improve/auto-enhance
Authorization: Bearer {STARGATE_API_KEY}
```
Triggers autonomous agents to create and load improvements automatically.

## 🎯 **Use Cases for Self-Improvement**

### 1. **Enhanced Scanning Algorithms**
```python
# Agent develops in sandbox
def enhanced_stego_detection(image_data):
    """New steganography detection using advanced ML"""
    # Agent-improved algorithm
    return improved_detection_result
```
Loads to: `/improved/scan/enhanced_stego_detection`

### 2. **Intelligent Agent Behavior**
```python  
# Agent develops better decision making
def smart_proposal_evaluation(proposal):
    """AI-enhanced proposal evaluation"""
    # Improved logic
    return better_decision
```
Loads to: `/improved/agent/smart_proposal_evaluation`

### 3. **API Extensions**
```python
# Agent develops new API capabilities
def blockchain_analysis(tx_data):
    """Advanced blockchain analysis"""
    # New functionality
    return insights
```
Loads to: `/improved/api/blockchain_analysis`

## 🔄 **Self-Improvement Workflow**

### Autonomous Cycle
1. **Analyze Current Performance**: Agents monitor system metrics
2. **Identify Improvement Areas**: Detection bottlenecks, accuracy gaps
3. **Develop Solutions**: Create improved functions in sandboxes
4. **Test & Validate**: Internal testing before loading
5. **Deploy**: Load via authenticated API
6. **Monitor**: Track improvement effectiveness

### Manual Enhancement
1. Agent creates improvement code
2. Operator reviews and approves
3. Load via `/self-improve/load` endpoint
4. System enhanced immediately

## 🛡️ **Security Isolation**

### Contract-Based Isolation
- Each `visible_pixel_hash` = isolated environment
- Functions can only access their sandbox directory
- Working directory changed before execution
- Memory separation between modules

### API Key Protection
- All self-improvement operations require `STARGATE_API_KEY`
- Write operations protected, read operations public
- Audit logging of all improvement activities

### Code Validation
- AST security parsing before loading
- Runtime execution in controlled environment
- Error isolation prevents system crashes

## 📊 **Monitoring Self-Improvement**

### System Status
```bash
curl http://localhost:8080/self-improve/status
```

Example Response:
```json
{
  "status": "active",
  "total_loaded": 5,
  "scan_improvements": [
    {
      "module_id": "abc123...",
      "function_name": "enhanced_detection",
      "endpoint_path": "/improved/scan/enhanced_detection",
      "loaded_at": "2024-01-28T10:30:00Z",
      "visible_pixel_hash": "def456..."
    }
  ],
  "agent_improvements": [...],
  "api_improvements": [...],
  "self_improvement_enabled": true,
  "agent_development_capable": true,
  "timestamp": "2024-01-28T10:35:00Z"
}
```

### Performance Tracking
- Each improvement tagged with load timestamp
- Endpoint performance monitored via existing metrics
- Success/failure rates tracked
- Rollback capability for problematic improvements

## 🚀 **Getting Started**

### 1. Enable Self-Improvement
```bash
# Set environment
export STARGATE_API_KEY="your-secure-api-key"
export UPLOADS_DIR="/data/uploads"

# Start enhanced Bitcoin API
python3 bitcoin_api.py
```

### 2. Create Agent Improvement
Agent creates code in sandbox:
```python
# /data/uploads/results/[hash]/my_improvements.py
def improved_algorithm(data):
    """Enhanced processing algorithm"""
    return enhanced_result
```

### 3. Load Improvement
```bash
curl -X POST "http://localhost:8080/self-improve/load" \
  -H "Authorization: Bearer your-secure-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "visible_pixel_hash": "abcdef1234567890...",
    "function_name": "improved_algorithm", 
    "module_name": "my_improvements",
    "improvement_type": "scan"
  }'
```

### 4. Use Enhanced Function
```bash
# New endpoint immediately available
curl "http://localhost:8080/improved/scan/improved_algorithm?data=..."
```

## 📈 **Benefits**

### Immediate Enhancement
- No system restarts required
- Zero-downtime improvements
- Real-time capability expansion

### Safety & Isolation
- Contract-based isolation prevents interference
- AST validation blocks dangerous code
- Rollback capability for problematic changes

### Autonomous Evolution
- Agents can self-improve based on performance
- Continuous capability enhancement
- Adaptive to new threats/requirements

### Audit & Control
- All improvements logged and tracked
- API key authentication prevents abuse
- Manual override capability

This integration transforms Starlight from a static system into an **autonomously evolving platform** that can adapt and enhance its own capabilities while maintaining security and reliability.