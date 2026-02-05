"""
Starlight Autonomous Agents Module
Stand-alone agent system that can be imported and run by other services.
This module contains the core agent logic extracted from run_agents.py
to make it reusable as an imported module.
"""

import time
import logging
import sys
import threading
from typing import List, Dict, Optional, Any
from .config import Config as AgentConfig
from .client import StargateClient
from .watcher import WatcherAgent
from .worker import WorkerAgent

# Configure logging
logger = logging.getLogger(__name__)

class AgentManager:
    """Manages autonomous agents with lifecycle control."""
    
    def __init__(self):
        self.client: Optional[StargateClient] = None
        self.watcher: Optional[WatcherAgent] = None
        self.worker: Optional[WorkerAgent] = None
        self.running = False
        self.agent_thread: Optional[threading.Thread] = None
        
        # Container-specific optimizations
        self.cycle_count = 0
        self.max_cycles = 10000  # High limit to prevent infinite loops in container
        
    def initialize(self, client: Optional[StargateClient] = None, 
                  ai_identifier: str = AgentConfig.AI_IDENTIFIER):
        """Initialize agents with optional client injection."""
        try:
            # Use provided client or create new one
            self.client = client or StargateClient()
            
            # Bind wallet if configured (needed for write operations like approval)
            if AgentConfig.DONATION_ADDRESS:
                logger.info(f"Attempting to bind wallet: {AgentConfig.DONATION_ADDRESS}")
                self.client.bind_wallet(AgentConfig.DONATION_ADDRESS)
            else:
                logger.warning("No DONATION_ADDRESS configured. Write operations might fail.")

            # Initialize agents
            self.watcher = WatcherAgent(self.client, ai_identifier)
            self.worker = WorkerAgent(self.client, ai_identifier)
            
            logger.info("AgentManager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AgentManager: {e}")
            return False
    
    def start(self, blocking: bool = False, max_cycles: Optional[int] = None):
        """Start the autonomous agents."""
        if self.running:
            logger.warning("Agents are already running")
            return False
            
        if not self.watcher or not self.worker:
            logger.error("Agents not initialized. Call initialize() first.")
            return False
            
        self.running = True
        self.max_cycles = max_cycles or self.max_cycles
        self.cycle_count = 0
        
        logger.info("Starting autonomous agents...")
        logger.info(f"Target Stargate URL: {AgentConfig.STARGATE_API_URL}")
        logger.info(f"AI Identifier: {AgentConfig.AI_IDENTIFIER}")
        
        try:
            if blocking:
                # Run in main thread
                self._run_agent_loop()
            else:
                # Run in background thread
                self.agent_thread = threading.Thread(
                    target=self._run_agent_loop, 
                    daemon=True
                )
                self.agent_thread.start()
                logger.info("Autonomous agents started in background thread")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start agents: {e}")
            self.running = False
            return False
    
    def stop(self, timeout: int = 10):
        """Stop the autonomous agents gracefully."""
        if not self.running:
            return True
            
        logger.info("Stopping autonomous agents...")
        self.running = False
        
        # Wait for agent thread to finish
        if self.agent_thread and self.agent_thread.is_alive():
            self.agent_thread.join(timeout=timeout)
            if self.agent_thread.is_alive():
                logger.warning("Agent thread did not stop gracefully")
            else:
                logger.info("Agent thread stopped successfully")
        
        # Save final state
        if self.watcher and self.worker:
            try:
                self.watcher._save_state()
                self.worker._save_state()
                logger.info("Final agent state saved")
            except Exception as e:
                logger.error(f"Failed to save final state: {e}")
        
        return True
    
    def _run_agent_loop(self):
        """Main agent execution loop."""
        try:
            while self.running and self.cycle_count < self.max_cycles:
                self.cycle_count += 1
                
                # Log cycle info periodically
                if self.cycle_count % 50 == 0:
                    logger.info(f"Agent cycle {self.cycle_count}/{self.max_cycles}")
                
                # 1. Worker looks for wishes and creates proposals
                self.worker.process_wishes()

                # 2. Watcher looks for proposals (audits/approves them) and tasks
                tasks = self.watcher.run_once()
                
                # 3. Worker processes available tasks
                for task in tasks:
                    self.worker.process_task(task)
                
                # 4. Wait with adaptive timing
                if not tasks:
                    logger.debug("No new tasks found, increasing poll interval")
                    time.sleep(AgentConfig.POLL_INTERVAL * 2)  # Slower when idle
                else:
                    time.sleep(AgentConfig.POLL_INTERVAL)
                    
                # Graceful exit check for container environment
                if self.cycle_count >= self.max_cycles:
                    logger.info("Reached maximum cycles, performing graceful shutdown...")
                    self.watcher._save_state()
                    self.worker._save_state()
                    break
                    
        except Exception as e:
            logger.error(f"Agent loop error: {e}")
        finally:
            self.running = False
            logger.info("Agent loop ended")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent system status."""
        return {
            "running": self.running,
            "cycle_count": self.cycle_count,
            "max_cycles": self.max_cycles,
            "agents_initialized": bool(self.watcher and self.worker),
            "client_connected": bool(self.client),
            "agent_thread_alive": bool(self.agent_thread.is_alive()) if self.agent_thread else False,
            "config": {
                "stargate_url": AgentConfig.STARGATE_API_URL,
                "ai_identifier": AgentConfig.AI_IDENTIFIER,
                "poll_interval": AgentConfig.POLL_INTERVAL,
                "donation_address": AgentConfig.DONATION_ADDRESS,
            }
        }
    
    def process_single_cycle(self) -> Dict[str, Any]:
        """Execute a single agent cycle and return results."""
        if not self.watcher or not self.worker:
            return {"error": "Agents not initialized"}
            
        try:
            self.cycle_count += 1
            
            # Execute one full cycle
            self.worker.process_wishes()
            tasks = self.watcher.run_once()
            
            processed_tasks = []
            for task in tasks:
                result = self.worker.process_task(task)
                processed_tasks.append({
                    "task_id": task.get("task_id"),
                    "success": bool(result),
                    "task": task
                })
            
            return {
                "cycle": self.cycle_count,
                "tasks_found": len(tasks),
                "tasks_processed": len(processed_tasks),
                "task_results": processed_tasks,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "cycle": self.cycle_count,
                "error": str(e),
                "timestamp": time.time()
            }

# Global instance for module-level access
_global_manager = AgentManager()

def get_manager() -> AgentManager:
    """Get the global agent manager instance."""
    return _global_manager

def start_agents(blocking: bool = False, max_cycles: Optional[int] = None, 
               client: Optional[StargateClient] = None) -> bool:
    """Convenience function to start agents."""
    return _global_manager.start(blocking=blocking, max_cycles=max_cycles)

def stop_agents(timeout: int = 10) -> bool:
    """Convenience function to stop agents."""
    return _global_manager.stop(timeout=timeout)

def get_agent_status() -> Dict[str, Any]:
    """Convenience function to get agent status."""
    return _global_manager.get_status()

def process_cycle() -> Dict[str, Any]:
    """Convenience function to process one cycle."""
    return _global_manager.process_single_cycle()

def initialize_agents(ai_identifier: str = AgentConfig.AI_IDENTIFIER,
                     client: Optional[StargateClient] = None) -> bool:
    """Convenience function to initialize agents."""
    return _global_manager.initialize(client=client, ai_identifier=ai_identifier)