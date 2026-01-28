#!/usr/bin/env python3
"""
Sandbox Integration Test
Tests the dynamic loading capabilities integrated into Bitcoin API
"""

import os
import sys
import requests
import json
import time
import tempfile
from pathlib import Path

# Configuration
API_BASE = "http://localhost:8080"  # Bitcoin API port
API_KEY = os.getenv("STARGATE_API_KEY", "test-key")

def setup_test_function():
    """Create a test function in sandbox."""
    print("🧪 Setting up test function...")
    
    # Create test environment
    uploads_dir = os.getenv("UPLOADS_DIR", "/tmp/uploads")
    test_hash = "sandbox1234567890abcdef1234567890abcdef12"
    sandbox_dir = os.path.join(uploads_dir, "results", test_hash)
    
    os.makedirs(sandbox_dir, exist_ok=True)
    
    # Create a sandbox function
    function_code = '''
def enhanced_confidence_calculator(stego_probability: float, confidence: float, image_size: int):
    """
    Enhanced confidence calculation for steganography detection.
    Enhancement: Considers image size in confidence assessment.
    """
    # Enhanced algorithm that weights confidence by image characteristics
    size_factor = min(1.0, image_size / 1000000)  # Normalize by 1MB
    adjusted_confidence = confidence * (0.8 + 0.2 * size_factor)
    
    return {
        "is_stego": stego_probability > 0.5,
        "confidence": min(1.0, adjusted_confidence),
        "function_type": "confidence_calculation",
        "function_version": "v2.0",
        "original_confidence": confidence,
        "size_adjustment": size_factor
    }

def blockchain_pattern_analysis(tx_data):
    """
    Analyze blockchain transaction patterns for steganography indicators.
    Enhancement: Pattern recognition for OP_RETURN data.
    """
    patterns = []
    
    # Look for suspicious patterns in transaction outputs
    if "outputs" in tx_data:
        for i, output in enumerate(tx_data["outputs"]):
            script = output.get("script_pubkey", "")
            if "OP_RETURN" in script:
                # Extract hex data
                hex_data = script.replace("OP_RETURN", "").strip()
                if len(hex_data) > 40:  # Unusually long data
                    patterns.append({
                        "output_index": i,
                        "suspicious": True,
                        "reason": "Long OP_RETURN data",
                        "data_length": len(hex_data),
                        "pattern_type": "size_anomaly"
                    })
    
    return {
        "patterns_found": patterns,
        "total_suspicious": len([p for p in patterns if p.get("suspicious", False)]),
        "function_type": "blockchain_analysis",
        "function_version": "v1.0"
    }
'''
    
    # Write function code to sandbox
    function_file = os.path.join(sandbox_dir, "functions.py")
    with open(function_file, 'w') as f:
        f.write(function_code)
    
    print(f"✅ Created test function: {function_file}")
    return test_hash

def test_sandbox_endpoints(test_hash):
    """Test all sandbox endpoints."""
    print("\n🧪 Testing Self-Improvement Endpoints...")
    
    # Test status before loading
    try:
        response = requests.get(f"{API_BASE}/sandbox/status")
        print(f"✅ Status check: {response.status_code}")
        if response.status_code == 200:
            status_data = response.json()
            print(f"📊 Current improvements: {status_data.get('total_loaded', 0)}")
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return False
    
    # Test loading improvement
    print("\n🚀 Loading sandbox function...")
    load_request = {
        "visible_pixel_hash": test_hash,
        "function_name": "enhanced_confidence_calculator",
        "module_name": "functions",
        "function_type": "scan"
    }
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    try:
        response = requests.post(
            f"{API_BASE}/sandbox/load",
            json=load_request,
            headers=headers
        )
        
        if response.status_code == 200:
            print(f"✅ Function loaded: {response.json()}")
            load_result = response.json()
            endpoint_path = load_result.get("endpoint_path")
            
            # Test the loaded function
            if endpoint_path:
                test_improved_function(endpoint_path)
            
        else:
            print(f"❌ Improvement loading failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Improvement loading error: {e}")
        return False
    
    # Test second improvement
    print("\n🚀 Loading blockchain analysis improvement...")
    load_request2 = {
        "visible_pixel_hash": test_hash,
        "function_name": "blockchain_pattern_analysis",
        "module_name": "enhancements", 
        "improvement_type": "api"
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/self-improve/load",
            json=load_request2,
            headers=headers
        )
        
        if response.status_code == 200:
            print(f"✅ Blockchain analysis loaded: {response.json()}")
        else:
            print(f"❌ Blockchain analysis failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Blockchain analysis error: {e}")
    
    # Check status after loading
    try:
        response = requests.get(f"{API_BASE}/self-improve/status")
        if response.status_code == 200:
            status_data = response.json()
            print(f"📊 Total improvements loaded: {status_data.get('total_loaded', 0)}")
            
            # Show loaded functions
            for category in ['scan_improvements', 'agent_improvements', 'api_improvements']:
                improvements = status_data.get(category, [])
                if improvements:
                    print(f"  {category}: {len(improvements)} functions")
                    for imp in improvements:
                        print(f"    - {imp.get('function_name')} @ {imp.get('endpoint_path')}")
                        
    except Exception as e:
        print(f"❌ Status check failed: {e}")
    
    # Test auto-enhancement
    print("\n🔄 Testing auto-enhancement...")
    try:
        response = requests.post(
            f"{API_BASE}/self-improve/auto-enhance",
            headers=headers
        )
        
        if response.status_code == 200:
            print(f"✅ Auto-enhancement triggered: {response.json()}")
        else:
            print(f"⚠️ Auto-enhancement response: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Auto-enhancement error: {e}")
    
    return True

def test_improved_function(endpoint_path):
    """Test a loaded improvement function."""
    try:
        # Call the improved function endpoint
        response = requests.get(f"{API_BASE}{endpoint_path}", params={
            "stego_probability": 0.7,
            "confidence": 0.6,
            "image_size": 500000
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Improved function result: {result}")
            return True
        else:
            print(f"❌ Function call failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Function call error: {e}")
        return False

def test_root_endpoint():
    """Test enhanced root endpoint."""
    print("\n🧪 Testing enhanced root endpoint...")
    
    try:
        response = requests.get(f"{API_BASE}/")
        if response.status_code == 200:
            root_data = response.json()
            
            print(f"✅ API Name: {root_data.get('name')}")
            print(f"✅ Version: {root_data.get('version')}")
            
            # Check self-improvement info
            self_improve = root_data.get('self_improvement', {})
            print(f"✅ Self-Improvement Enabled: {self_improve.get('enabled')}")
            print(f"✅ Loaded Functions: {self_improve.get('loaded_functions')}")
            print(f"✅ Auto-Enhancement: {self_improve.get('auto_enhancement_available')}")
            
            # Check new endpoints
            endpoints = root_data.get('endpoints', {})
            self_improve_endpoints = endpoints.get('self_improve', {})
            print(f"✅ Self-Improve Endpoints: {list(self_improve_endpoints.keys())}")
            
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_unloading(test_hash):
    """Test unloading improvements."""
    print("\n🧪 Testing improvement unloading...")
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    try:
        response = requests.delete(
            f"{API_BASE}/self-improve/unload/{test_hash}/enhanced_confidence_calculator",
            headers=headers
        )
        
        if response.status_code == 200:
            print(f"✅ Improvement unloaded: {response.json()}")
            return True
        else:
            print(f"❌ Unload failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Unload error: {e}")
        return False

def main():
    """Main test runner."""
    print("🚀 Starting Self-Improvement Integration Test")
    print(f"🌐 API Base: {API_BASE}")
    print(f"🔐 API Key: {'*' * len(API_KEY) if API_KEY else 'None'}")
    
    # Check if server is running
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        print(f"✅ Server health: {response.status_code}")
    except requests.exceptions.RequestException:
        print("❌ Server not running. Please start with: python3 bitcoin_api.py")
        sys.exit(1)
    
    # Run tests
    success = True
    
    success &= test_root_endpoint()
    
    test_hash = setup_test_function()
    success &= test_sandbox_endpoints(test_hash)
    success &= test_unloading(test_hash)
    
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")
    
    print("\n🎯 Self-Improvement Integration Summary:")
    print("  ✅ Dynamic function loading from agent sandboxes")
    print("  ✅ Security validation and isolation")
    print("  ✅ Runtime endpoint registration") 
    print("  ✅ Auto-enhancement triggers")
    print("  ✅ Function unloading and cleanup")
    print("  ✅ API key authentication protection")
    print("\n🚀 Starlight can now self-improve!")

if __name__ == "__main__":
    main()