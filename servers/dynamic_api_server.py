#!/usr/bin/env python3
"""
Dynamic API Server
Example server demonstrating dynamic function loading from agent sandboxes
"""

import os
import uvicorn
from fastapi import FastAPI

from starlight.agents.dynamic_api import create_dynamic_app, setup_dynamic_routes

def main():
    """Main server entry point."""
    # Create the base FastAPI app
    app = create_dynamic_app()
    
    # Get configuration
    host = os.getenv("DYNAMIC_API_HOST", "0.0.0.0")
    port = int(os.getenv("DYNAMIC_API_PORT", "8000"))
    
    print(f"🚀 Starting Starlight Dynamic API Server")
    print(f"📍 Server will be available at: http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔐 Write operations require STARGATE_API_KEY authentication")
    print(f"📁 Sandbox directory: {os.getenv('UPLOADS_DIR', '/data/uploads')}")
    
    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT", "production") == "development",
        log_level="info"
    )

if __name__ == "__main__":
    main()