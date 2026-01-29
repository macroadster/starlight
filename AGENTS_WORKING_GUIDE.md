# AI Agent Working Guide - Project Starlight

**You are working in an isolated sandbox environment for Project Starlight.**

## 🚀 **Your Current Context**

### Project Overview
**Project Starlight** is an open-source protocol to build and train AI models for detecting steganography in images stored on blockchains like Bitcoin.

- **Primary Goal**: Safeguard integrity of digital history stored on-chain
- **Long-Term Vision (2142)**: Automate covert data detection for "AI common sense"
- **Your Mission**: Complete assigned tasks efficiently and securely

### Your Current Location
```bash
/data/uploads/results/[visible_pixel_hash]/
```
This is your isolated workspace where you should:
- Write all code and files
- Store your work output
- Test implementations
- Create deliverables

## 🛡️ **Security & Constraints**

### ✅ **Allowed Operations**
```python
# Safe imports you can use
import json, math, base64, hashlib, datetime, re, string
import itertools, collections, dataclasses
from typing import Dict, List, Optional, Any, Union

# Safe operations
math.sqrt(16)                    # Math operations
json.loads(data)                 # JSON parsing
base64.b64encode(data)          # Encoding
hashlib.sha256(data).hexdigest() # Hashing
datetime.datetime.now()           # Timestamps
re.findall(pattern, text)        # Regex
```

### ❌ **Blocked Operations**
```python
# These will be blocked by security validation
open()                          # File access
subprocess.run()                 # System commands  
socket.socket()                 # Network access
requests.get()                  # HTTP requests
eval() / exec()                 # Code execution
import os, sys, subprocess       # System imports
import requests, urllib, socket   # Network imports
```

### 🔒 **Isolation Rules**
- **Working Directory**: Limited to your sandbox only
- **File Access**: Cannot access files outside sandbox
- **Network**: No external network access
- **Execution**: Only allowed imports and operations

## 📁 **Project Structure Reference**

### Key Files (for context)
```bash
scanner.py                    # Main steganography detection tool
diag.py                      # Dataset integrity verification
trainer.py                    # Model training
datasets/[name]_submission_[year]/  # Dataset contributions
models/                       # Trained models
```

### Core Commands (for context)
```bash
# Dataset generation
cd datasets/<contributor>
python3 data_generator.py --limit 10

# Verify data integrity  
python3 diag.py

# Run detection
python3 scanner.py /path/to/image.png --json
```

## 🎯 **Your Task Workflow**

### 1. **Understand Your Assignment**
You'll receive a task description like:
```
TASK: Complete this work efficiently and provide concrete results.

REQUIREMENTS:
1. Provide specific implementation details
2. Include actual code examples or execution steps  
3. Show evidence of completion
4. Keep response concise and actionable
```

### 2. **Implementation Pattern**
```python
def solve_task(task_input):
    """
    Skill: Task-specific implementation
    Type: [analysis/processing/integration]
    Version: 1.0
    Author: [your_identifier]
    
    Args:
        task_input: Task parameters and context
        
    Returns:
        dict: Structured result with implementation
    """
    try:
        # Your solution logic here
        result = implement_solution(task_input)
        
        return {
            "success": True,
            "result": result,
            "error": None,
            "metadata": {
                "task_completed": True,
                "implementation_type": "direct",
                "completion_time": datetime.datetime.now().isoformat()
            }
        }
    except Exception as e:
        return {
            "success": False, 
            "result": None,
            "error": str(e),
            "metadata": {"task_completed": False}
        }
```

### 3. **Required Deliverables**
Always include:
- **Implementation details**: Code, logic, approach
- **Evidence of completion**: Test results, outputs, verification
- **Working files**: Any code created in your sandbox
- **Summary**: Clear status and results

## 🧪 **Testing & Verification**

### Local Testing
```python
def test_implementation():
    """Test your work before submission."""
    test_cases = [
        {"input": "test1", "expected": "result1"},
        {"input": "test2", "expected": "result2"}
    ]
    
    for case in test_cases:
        result = solve_task(case["input"])
        if result["success"]:
            print(f"✅ Test passed: {case['input']}")
        else:
            print(f"❌ Test failed: {result['error']}")
    
    return True

if __name__ == "__main__":
    test_implementation()
```

### Verification Checklist
- [ ] Code runs without errors
- [ ] Security constraints respected
- [ ] All deliverables present
- [ ] Clear documentation provided
- [ ] Evidence of completion

## 📝 **Common Task Types**

### Type 1: Analysis Tasks
```python
def analyze_data(data):
    """Analyze steganography patterns in image data."""
    patterns_found = []
    
    # Pattern detection logic
    if detect_anomalies(data):
        patterns_found.append("anomaly_detected")
    
    return {
        "analysis_complete": True,
        "patterns_found": patterns_found,
        "confidence": 0.85
    }
```

### Type 2: Processing Tasks  
```python
def process_dataset(raw_data):
    """Process and normalize dataset."""
    processed = []
    
    for item in raw_data:
        normalized = normalize_item(item)
        processed.append(normalized)
    
    return {
        "processed_items": len(processed),
        "data": processed,
        "processing_complete": True
    }
```

### Type 3: Implementation Tasks
```python
def implement_feature(requirements):
    """Implement new feature based on requirements."""
    
    # Code implementation
    feature_code = write_feature_code(requirements)
    
    # Test implementation
    test_results = test_feature(feature_code)
    
    return {
        "feature_implemented": True,
        "code_files": feature_code,
        "test_passed": test_results["success"],
        "implementation": feature_code
    }
```

## 🔄 **Work Completion Process**

### When Your Work is Done:
1. **Final verification**: Test everything works
2. **Documentation**: Ensure all code is documented
3. **Submit**: Your work will be automatically collected
4. **Audit**: Watcher will verify your deliverables

### Submission Format
```python
{
    "notes": "# Task Report\n\n## Implementation\n[Your work description]\n\n## Results\n[Evidence of completion]",
    "result_file": "/uploads/results/[hash]/[task_id].md",
    "artifacts_dir": "/uploads/results/[hash]/", 
    "completion_proof": "unique-identifier"
}
```

## 🚨 **Important Reminders**

### Security First
- Never attempt file system access outside sandbox
- Use only allowed imports and operations
- Handle all exceptions gracefully

### Quality Standards  
- Provide working, testable solutions
- Include clear documentation
- Show evidence of completion
- Follow the specific task requirements

### Communication
- Be concise and technical
- Focus on implementation details
- Provide concrete evidence
- Avoid conversational filler

## 🆘 **Getting Help**

If you encounter issues:
1. **Check constraints**: Ensure you're not using blocked operations
2. **Review requirements**: Verify you're meeting all task criteria  
3. **Test locally**: Verify your code works before submission
4. **Document**: Clearly explain any challenges and solutions

---

**You are ready to work in your Starlight sandbox! Focus on secure, efficient implementation of your assigned tasks.**