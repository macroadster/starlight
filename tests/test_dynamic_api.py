#!/usr/bin/env python3
"""
Test script for Dynamic API functionality
Demonstrates loading and calling agent-developed functions
"""

import os
import sys
import requests
import json
import time
from pathlib import Path

# Configuration
API_BASE = "http://localhost:8000"
API_KEY = os.getenv("STARGATE_API_KEY", "test_key")

def setup_test_sandbox():
    """Create a test sandbox with a sample function."""
    print("🧪 Setting up test sandbox...")
    
    # Create test directories
    uploads_dir = os.getenv("UPLOADS_DIR", "/tmp/uploads")
    test_hash = "abcdef1234567890abcdef1234567890abcdef12"
    sandbox_dir = os.path.join(uploads_dir, "results", test_hash)
    
    os.makedirs(sandbox_dir, exist_ok=True)
    
    # Create a test function
    test_function = '''
def hello_world():
    """Simple test function that returns a greeting."""
    return {"message": "Hello from agent sandbox!", "timestamp": str(int(time.time()))}

def add_numbers(a: int, b: int):
    """Another test function that adds two numbers."""
    return {"result": a + b, "operation": f"{a} + {b}"}
'''
    
    # Write test function to sandbox
    test_file = os.path.join(sandbox_dir, "test_functions.py")
    with open(test_file, 'w') as f:
        f.write(test_function)
    
    print(f"✅ Created test sandbox: {sandbox_dir}")
    print(f"📝 Created test file: {test_file}")
    
    return test_hash

def test_api_endpoints():
    """Test all API endpoints."""
    print("\n🧪 Testing API endpoints...")
    
    # Test health
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"✅ Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test list functions (should be empty initially)
    try:
        response = requests.get(f"{API_BASE}/list-functions")
        print(f"✅ List functions: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ List functions failed: {e}")
        return False
    
    return True

def test_function_loading(test_hash):
    """Test loading a function from sandbox."""
    print("\n🧪 Testing function loading...")
    
    load_request = {
        "visible_pixel_hash": test_hash,
        "function_name": "hello_world",
        "module_name": "test_functions",
        "endpoint_path": "/test/hello",
        "method": "GET"
    }
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    try:
        response = requests.post(
            f"{API_BASE}/load-function",
            json=load_request,
            headers=headers
        )
        
        if response.status_code == 200:
            print(f"✅ Function loaded successfully: {response.json()}")
            return True
        else:
            print(f"❌ Function loading failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Function loading error: {e}")
        return False

def test_function_calling(test_hash):
    """Test calling loaded functions."""
    print("\n🧪 Testing function calling...")
    
    # Test via dynamic endpoint
    try:
        response = requests.get(f"{API_BASE}/test/hello")
        if response.status_code == 200:
            print(f"✅ Dynamic endpoint call: {response.json()}")
        else:
            print(f"❌ Dynamic endpoint call failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Dynamic endpoint call error: {e}")
    
    # Test via function endpoint
    try:
        response = requests.get(f"{API_BASE}/function/{test_hash}/hello_world")
        if response.status_code == 200:
            print(f"✅ Function endpoint call: {response.json()}")
        else:
            print(f"❌ Function endpoint call failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Function endpoint call error: {e}")

def test_unauthorized_access():
    """Test that unauthorized access is blocked."""
    print("\n🧪 Testing unauthorized access protection...")
    
    load_request = {
        "visible_pixel_hash": "fake_hash",
        "function_name": "fake_function"
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/load-function",
            json=load_request
        )
        
        if response.status_code == 403:
            print("✅ Unauthorized access correctly blocked")
            return True
        else:
            print(f"❌ Unauthorized access not blocked: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Unauthorized test error: {e}")
        return False

def main():
    """Main test runner."""
    print("🚀 Starting Dynamic API Test Suite")
    print(f"🌐 API Base: {API_BASE}")
    print(f"🔐 API Key: {'*' * len(API_KEY) if API_KEY else 'None'}")
    
    # Check if server is running
    try:
        requests.get(f"{API_BASE}/health", timeout=5)
    except requests.exceptions.RequestException:
        print("❌ Server not running. Please start with: python3 dynamic_api_server.py")
        sys.exit(1)
    
    # Setup test environment
    test_hash = setup_test_sandbox()
    
    # Run tests
    success = True
    
    success &= test_api_endpoints()
    success &= test_unauthorized_access()
    
    if success:
        success &= test_function_loading(test_hash)
        test_function_calling(test_hash)
        
        # Test second function
        print("\n🧪 Loading second function...")
        load_request = {
            "visible_pixel_hash": test_hash,
            "function_name": "add_numbers",
            "module_name": "test_functions",
            "endpoint_path": "/test/add",
            "method": "POST"
        }
        
        headers = {"Authorization": f"Bearer {API_KEY}"}
        
        try:
            response = requests.post(
                f"{API_BASE}/load-function",
                json=load_request,
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Second function loaded successfully")
                
                # Test it
                response = requests.post(
                    f"{API_BASE}/test/add",
                    json={"a": 5, "b": 3}
                )
                print(f"✅ Second function call: {response.json()}")
        except Exception as e:
            print(f"❌ Second function test failed: {e}")
    
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")
    
    # Final cleanup
    print("\n🧹 Cleanup note: Test sandbox files remain in /tmp/uploads for manual inspection")

if __name__ == "__main__":
    main()