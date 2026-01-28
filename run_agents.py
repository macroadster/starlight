#!/usr/bin/env python3
import time
import logging
import sys
import os

# Add the current directory to path so we can import starlight modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from starlight.agents.config import Config
from starlight.agents.client import StargateClient
from starlight.agents.watcher import WatcherAgent
from starlight.agents.worker import WorkerAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("System")

def main():
    logger.info("Starting Starlight Autonomous Agent System...")
    logger.info(f"Target Stargate URL: {Config.STARGATE_API_URL}")
    logger.info(f"AI Identifier: {Config.AI_IDENTIFIER}")

    client = StargateClient()
    
    # Bind wallet if configured (needed for write operations like approval)
    if Config.DONATION_ADDRESS:
        logger.info(f"Attempting to bind wallet: {Config.DONATION_ADDRESS}")
        client.bind_wallet(Config.DONATION_ADDRESS)
    else:
        logger.warning("No DONATION_ADDRESS configured. Write operations might fail.")

    watcher = WatcherAgent(client)
    worker = WorkerAgent(client)

    logger.info("Agents initialized. Entering main loop.")

    try:
        while True:
            # 1. Worker looks for wishes and creates proposals
            worker.process_wishes()

            # 2. Watcher looks for proposals (approves them) and tasks
            tasks = watcher.run_once()
            
            # 3. Worker processes available tasks
            for task in tasks:
                worker.process_task(task)
            
            # 4. Wait
            if not tasks:
                pass # logger.debug("No new tasks found.")
            
            time.sleep(Config.POLL_INTERVAL)

    except KeyboardInterrupt:
        logger.info("Shutting down agents...")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Unexpected system crash: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
