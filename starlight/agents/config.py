import os

class Config:
    STARGATE_API_URL = os.getenv("STARGATE_API_URL", "http://stargate-backend:3001")
    STARGATE_API_KEY = os.getenv("STARGATE_API_KEY", "demo-api-key")
    AI_IDENTIFIER = os.getenv("AI_IDENTIFIER", "starlight-autonomous-agent")
    POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
    DONATION_ADDRESS = os.getenv("STARLIGHT_DONATION_ADDRESS", "")
    UPLOADS_DIR = os.getenv("UPLOADS_DIR", "/data/uploads")
