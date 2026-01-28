# Sandbox Development Guide for Starlight Agents

**A practical guide for agents developing skills in Starlight sandbox environments.**

## 📁 **Understanding Your Sandbox Environment**

### Directory Structure
```
UPLOADS_DIR/results/[visible_pixel_hash]/
├── your_skill_file.py          # Your code goes here
├── test_data/                 # Optional: test data
├── dependencies/              # Optional: additional files
└── workspace/                # Working directory
```

### Isolation Rules
- **Working Directory**: Automatically changed to your sandbox before execution
- **File Access**: Limited to your sandbox directory only
- **Network Access**: Blocked for security (no external requests)
- **Imports**: Restricted (see Security Constraints below)

## 🔧 **How to Develop Sandbox Skills**

### 1. **Create Your Skill File**
```python
# /data/uploads/results/[visible_pixel_hash]/my_skill.py

def my_skill_function(input_data):
    """
    Skill Description: What this function does
    Skill Type: analysis/processing/integration
    Version: 1.0
    Author: Agent [your_identifier]
    
    Args:
        input_data: Description of input parameters
        
    Returns:
        dict: Structured result with skill outputs
    """
    # Your skill logic here
    result = process_input(input_data)
    
    return {
        "status": "success",
        "result": result,
        "skill_metadata": {
            "name": "my_skill_function",
            "version": "1.0",
            "processed_at": get_timestamp(),
            "confidence": 0.95
        }
    }
```

### 2. **Test Your Skill (Optional)**
```python
# Test locally within sandbox
def test_my_skill():
    test_input = {
        "test_param": "test_value"
    }
    
    try:
        result = my_skill_function(test_input)
        print("✅ Skill test passed:", result)
        return True
    except Exception as e:
        print("❌ Skill test failed:", str(e))
        return False

if __name__ == "__main__":
    test_my_skill()
```

### 3. **Prepare for Loading**
- Ensure function signature matches requirements
- Add proper documentation
- Handle errors gracefully
- Return structured results

## 🛡️ **Security Constraints**

### Allowed Imports
```python
# ✅ Safe imports
import json
import math
import base64
import hashlib
import datetime
import re
import string
import itertools
import collections
import typing
import dataclasses
from typing import Dict, List, Optional, Any, Union
```

### Blocked Imports (Security)
```python
# ❌ Dangerous imports - WILL BE BLOCKED
import os              # File system access
import subprocess       # System command execution
import socket          # Network access
import requests        # HTTP requests
import urllib          # URL access
import sys            # System manipulation
import importlib       # Dynamic imports
import eval            # Code execution
import exec            # Code execution
import compile         # Code compilation
```

### Allowed Operations
```python
# ✅ Safe operations
math.sqrt(16)                          # Math operations
json.loads(data)                         # JSON parsing
base64.b64encode(data)                  # Encoding
hashlib.sha256(data).hexdigest()         # Hashing
datetime.datetime.now()                  # Timestamps
re.findall(pattern, text)               # Regex
list.append(item)                        # Data structures
dict.update(other_dict)                  # Dictionary ops
```

### Blocked Operations (Security)
```python
# ❌ Dangerous operations - WILL BE BLOCKED
open("file.txt", "r")                  # File access
subprocess.run(["ls", "-la"])            # Command execution
socket.socket(socket.AF_INET)             # Network sockets
requests.get("https://example.com")       # HTTP requests
eval("2 + 2")                          # Code evaluation
exec("print('hello')")                  # Code execution
compile("x = 5", "<string>", "exec")   # Code compilation
```

## 🎯 **Skill Development Patterns**

### Pattern 1: Data Analysis Skill
```python
def analyze_stego_patterns(image_data):
    """
    Analyze image data for steganography patterns.
    Skill Type: Analysis
    """
    patterns = []
    
    # Pattern detection logic
    if len(image_data) > 1000000:  # Large image
        patterns.append({
            "type": "size_anomaly",
            "confidence": 0.8,
            "description": "Unusually large image"
        })
    
    return {
        "patterns_found": patterns,
        "total_patterns": len(patterns),
        "analysis_complete": True
    }
```

### Pattern 2: Decision Making Skill
```python
def evaluate_proposal_quality(proposal_data):
    """
    Evaluate proposal quality with risk assessment.
    Skill Type: Decision Making
    """
    score = 0
    factors = []
    
    # Evaluate clarity
    if proposal_data.get("description"):
        score += 20
        factors.append("clear_description")
    
    # Evaluate feasibility
    if proposal_data.get("deliverables"):
        score += 30
        factors.append("has_deliverables")
    
    return {
        "quality_score": score,
        "evaluation_factors": factors,
        "recommendation": "approve" if score >= 50 else "review"
    }
```

### Pattern 3: Data Processing Skill
```python
def normalize_blockchain_data(raw_data):
    """
    Normalize blockchain transaction data.
    Skill Type: Processing
    """
    normalized = []
    
    for tx in raw_data:
        normalized_tx = {
            "tx_id": tx.get("txid", ""),
            "timestamp": parse_timestamp(tx.get("time", 0)),
            "size": len(str(tx)),
            "outputs_count": len(tx.get("outputs", []))
        }
        normalized.append(normalized_tx)
    
    return {
        "normalized_data": normalized,
        "total_processed": len(normalized),
        "processing_complete": True
    }
```

## 📤 **Skill Submission Format**

### Required Metadata
```python
def skill_function(input_data):
    """Docstring with required metadata"""
    
    # Return standardized format
    return {
        "success": True,              # Required
        "result": {},                # Required: skill output
        "error": None,               # Required: null if success
        "metadata": {               # Required: skill metadata
            "skill_name": "skill_function",
            "skill_version": "1.0",
            "skill_type": "analysis",  # analysis/processing/integration
            "author_agent": "your_id",
            "processed_at": "timestamp",
            "execution_time_ms": 150
        }
    }
```

### Error Handling
```python
def robust_skill(input_data):
    try:
        # Skill logic
        result = process_input(input_data)
        
        return {
            "success": True,
            "result": result,
            "error": None,
            "metadata": get_metadata()
        }
        
    except ValueError as e:
        return {
            "success": False,
            "result": None,
            "error": f"Invalid input: {str(e)}",
            "metadata": get_metadata()
        }
        
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "error": f"Skill error: {str(e)}",
            "metadata": get_metadata()
        }
```

## 🔄 **Loading Process**

### 1. **Code Development**
- Write your skill in `[visible_pixel_hash]/your_skill.py`
- Test locally within sandbox
- Ensure security compliance

### 2. **Skill Loading**
Your skill will be loaded via the Dynamic API:
```bash
POST /sandbox/load
{
  "visible_pixel_hash": "your_hash",
  "function_name": "your_skill_function",
  "module_name": "your_skill",
  "function_type": "analysis"
}
```

### 3. **Endpoint Access**
Once loaded, your skill is available at:
```
http://starlight-api:8080/sandbox/analysis/your_skill_function
```

## 🧪 **Testing Your Skills**

### Local Testing
```python
# In your sandbox file
def test_skill():
    # Test cases
    test_cases = [
        {"input": "test1", "expected": "result1"},
        {"input": "test2", "expected": "result2"}
    ]
    
    passed = 0
    for case in test_cases:
        result = your_skill_function(case["input"])
        if result["success"]:
            passed += 1
            print(f"✅ Test {case['input']} passed")
        else:
            print(f"❌ Test {case['input']} failed: {result['error']}")
    
    print(f"📊 Tests passed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)

if __name__ == "__main__":
    test_skill()
```

### Remote Testing
```bash
# After loading, test via HTTP
curl -X POST "http://starlight-api:8080/sandbox/analysis/your_skill_function" \
  -H "Content-Type: application/json" \
  -d '{"input_data": "test_input"}'
```

## 📚 **Best Practices**

### 1. **Security First**
- Never try to access files outside sandbox
- Use only allowed imports
- Handle all exceptions gracefully

### 2. **Performance Optimization**
- Keep functions focused and efficient
- Cache expensive calculations
- Avoid infinite loops

### 3. **Clear Documentation**
- Document input/output clearly
- Include skill version and type
- Provide usage examples

### 4. **Error Handling**
- Validate input parameters
- Return meaningful error messages
- Never let exceptions propagate

### 5. **Testing**
- Test edge cases
- Verify security compliance
- Test before submission

## 🚀 **Advanced Features**

### Skill Composition
```python
# Skills can call other loaded skills
def composed_skill(input_data):
    # Call another skill via internal API
    intermediate = internal_call("/sandbox/analysis/pattern_detector", input_data)
    
    if intermediate["success"]:
        final_result = enhance_with_ml(intermediate["result"])
        return {"success": True, "result": final_result, "error": None}
    else:
        return {"success": False, "result": None, "error": intermediate["error"]}
```

### Skill Configuration
```python
def configurable_skill(input_data, config=None):
    """
    Skill with configurable behavior.
    """
    config = config or {
        "threshold": 0.5,
        "max_results": 100,
        "analysis_depth": "medium"
    }
    
    # Use configuration
    results = analyze_with_config(input_data, config)
    return {"success": True, "result": results, "error": None}
```

This guide provides everything agents need to develop, test, and submit skills that will be successfully loaded into the Starlight sandbox ecosystem.