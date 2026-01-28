# Agent Sandbox Quick Reference

**Fast reference for agents working in Starlight sandbox environments.**

## 📁 **Your Sandbox Location**
```
/data/uploads/results/[visible_pixel_hash]/
```
Your isolated directory where you can write files and develop skills.

## 🔧 **Skill Template**
```python
def skill_name(input_data):
    """
    [Skill Description]
    Type: [analysis/processing/integration]
    Version: 1.0
    """
    try:
        # Your skill logic here
        result = process_input(input_data)
        
        return {
            "success": True,
            "result": result,
            "error": None,
            "metadata": {
                "skill_name": "skill_name",
                "version": "1.0",
                "processed_at": str(datetime.datetime.now())
            }
        }
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "error": str(e),
            "metadata": {
                "skill_name": "skill_name",
                "version": "1.0"
            }
        }
```

## ✅ **Allowed Imports**
```python
import json, math, base64, hashlib, datetime, re, string
from typing import Dict, List, Optional, Any, Union
import itertools, collections, dataclasses
```

## ❌ **Blocked Operations**
```python
# These will be blocked by security validation
open()          # File access
subprocess.run() # System commands
socket.socket() # Network access
requests.get()   # HTTP requests
eval()          # Code execution
exec()          # Code execution
```

## 📤 **Loading Your Skill**
Once you create your skill file, it can be loaded:
```bash
POST /sandbox/load
{
  "visible_pixel_hash": "your_hash",
  "function_name": "skill_name",
  "module_name": "your_file_name",
  "function_type": "analysis"
}
```

## 🔗 **Access Other Skills**
Loaded skills can call each other:
```python
# After loading, access at:
# http://starlight-api:8080/sandbox/[type]/[skill_name]
```

## 📊 **Status Monitoring**
```bash
# Check all loaded skills
GET /sandbox/status
```

## 🛡️ **Security Rules**
- Limited to your sandbox directory
- AST validation blocks dangerous code
- Network access blocked
- File system access restricted

## 📝 **Development Tips**
1. **Test locally** before submission
2. **Handle errors** gracefully
3. **Document** inputs/outputs clearly
4. **Return structured** data with metadata
5. **Keep functions** focused and efficient

## 🚀 **Getting Help**
- Full guide: `AGENT_SANDBOX_GUIDE.md`
- API docs: `http://localhost:8080/docs`
- Support: Check agent logs

Your sandbox is ready for skill development!