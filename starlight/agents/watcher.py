import logging
import time
import shutil
from typing import List, Dict, Set, Optional, Any, Tuple
from .client import StargateClient
from .config import Config
from .opencode_client import OpenCodeMCPClient

logger = logging.getLogger(__name__)

class WatcherAgent:
    def __init__(self, client: StargateClient, ai_identifier: str = Config.AI_IDENTIFIER):
        self.client = client
        self.ai_identifier = ai_identifier
        self.opencode_client = OpenCodeMCPClient()
        
        # In-memory caches for current session (will reset on restart)
        self.seen_tasks: Set[str] = set()
        self.seen_proposals: Set[str] = set()
        self.seen_submissions: Set[str] = set()
        self.rejected_proposals: Set[str] = set()
        self.rejection_cache: Dict[str, str] = {}  # proposal_id -> rejection_reason
        
        # Persistent storage key for this agent instance
        self.storage_key = f"watcher_state_{ai_identifier.lower().replace('-', '_')}"
        
        # Load persisted state from MCP storage
        self._load_state()
        
        # Disk space monitoring
        self.last_cleanup_time = 0
        self.cleanup_interval = 3600  # 1 hour between cleanups
        
        # Keep subprocess fallback for compatibility
        self.opencode_path = shutil.which("opencode")
        if self.opencode_client.is_available():
            logger.info("Watcher: OpenCode MCP client connected. Efficient auditing enabled.")
        elif self.opencode_path:
            logger.info(f"Watcher: OpenCode detected at {self.opencode_path}. Subprocess auditing enabled.")
        else:
            logger.warning("Watcher: OpenCode not found. Falling back to auto-approval.")

    def _is_recursive_proposal(self, proposal: Dict) -> bool:
        """Detect if a proposal is just creating another contract/proposal instead of doing work."""
        title = proposal.get("title", "").lower()
        desc = proposal.get("description_md", "").lower()
        
        # RED FLAGS: Indicators of recursive/proxy proposals
        recursive_indicators = [
            "create a proposal",
            "submit a proposal", 
            "make a proposal",
            "generate proposal",
            "write a proposal",
            "proposal for the proposal",
            "create another proposal",
            "build a proposal",
            "draft a proposal"
        ]
        
        # Check title and description for recursive patterns
        for indicator in recursive_indicators:
            if indicator in title or indicator in desc:
                logger.warning(f"Recursive proposal detected: '{indicator}' found in title/desc")
                return True
        
        # Check if proposal is just restating the wish without implementation
        if "fulfill the wish" in desc and len(desc.split()) < 50:
            logger.warning("Shallow proposal detected: Just restates wish without implementation details")
            return True
            
        # Check if proposal creates tasks that are just "create proposal"
        tasks = proposal.get("tasks", [])
        for task in tasks:
            if isinstance(task, dict):
                task_desc = task.get("description", "").lower()
                task_title = task.get("title", "").lower()
                if any(indicator in task_desc or indicator in task_title for indicator in recursive_indicators):
                    logger.warning("Recursive task detected: Task is just to create another proposal")
                    return True
        
        return False

    def _notify_worker_of_rejection(self, proposal_id: str, rejection_reason: str):
        """Store rejection information for worker to access during revisions."""
        try:
            # Store rejection in shared file location that worker can access
            rejection_info = {
                "proposal_id": proposal_id,
                "rejection_reason": rejection_reason,
                "rejected_by": self.ai_identifier,
                "timestamp": time.time()
            }
            
            # File storage only (MCP not available)
            import json
            rejection_file = f"rejection_{proposal_id}.json"
            with open(rejection_file, 'w') as f:
                json.dump(rejection_info, f)
            logger.debug(f"Stored rejection info for worker in file: {rejection_file}")
            
        except Exception as e:
            logger.warning(f"Failed to store rejection info for worker: {e}")

    def find_available_tasks(self) -> List[Dict]:
        """Scans for available or already claimed tasks in active/approved proposals."""
        available_tasks = []
        proposals = self.client.get_proposals()
        
        if proposals is None:
            logger.warning("Watcher: find_available_tasks received None from get_proposals")
            return []

        for proposal in proposals:
            if not isinstance(proposal, dict):
                logger.warning(f"Watcher: find_available_tasks received non-dict proposal: {proposal}")
                continue

            # ONLY look for tasks in proposals that are already approved/active
            p_status = proposal.get("status", "").lower()
            if p_status not in ["active", "approved", "published"]:
                continue

            tasks = proposal.get("tasks", [])
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                task_id = task.get("task_id")
                status = task.get("status", "").lower()
                claimed_by = task.get("claimed_by", "").lower()
                
                # surfacing: 1. available tasks, or 2. tasks claimed by us (for resumption)
                # Check both identifier and wallet address
                is_ours = claimed_by == self.ai_identifier.lower()
                if not is_ours and Config.DONATION_ADDRESS:
                    is_ours = claimed_by == Config.DONATION_ADDRESS.lower()
                
                if (status == "available" or (status == "claimed" and is_ours)) and task_id and task_id not in self.seen_tasks:
                    task["proposal_id"] = proposal.get("id")
                    task["proposal_title"] = proposal.get("title")
                    available_tasks.append(task)
                    self.seen_tasks.add(task_id)
        
        if available_tasks:
            logger.info(f"Watcher found {len(available_tasks)} actionable tasks (available or resuming).")
            
        return available_tasks

    def process_pending_proposals(self):
        """Finds and approves pending proposals after auditing them."""
        proposals = self.client.get_proposals()
        if proposals is None:
            return
            
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            pid = proposal.get("id")
            status = proposal.get("status", "").lower()
            
            if status == "pending" and pid not in self.seen_proposals:
                # Check if already rejected (avoid re-auditing)
                if pid in self.rejected_proposals:
                    logger.debug(f"Watcher: Proposal {pid} already rejected. Skipping audit.")
                    self.seen_proposals.add(pid)  # Mark as seen to avoid rechecking
                    continue
                    
                logger.info(f"Watcher auditing pending proposal {pid}...")
                
                if self._audit_proposal(proposal):
                    logger.info(f"Proposal {pid} passed audit. Approving...")
                    if pid and self.client.approve_proposal(pid):
                        logger.info(f"Proposal {pid} approved.")
                        self.seen_proposals.add(pid)
                else:
                    logger.warning(f"Proposal {pid} failed audit. Marking as rejected.")
                    if pid:
                        self.rejected_proposals.add(pid)
                        self.seen_proposals.add(pid)  # Also add to seen to avoid reprocessing
                        rejection_reason = self.rejection_cache.get(pid, "Failed audit")
                        logger.info(f"Watcher: Rejected proposal {pid}: {rejection_reason}")
                        
                        # Notify worker about rejection for revision purposes
                        self._notify_worker_of_rejection(pid, rejection_reason)

    def _audit_proposal(self, proposal: Dict) -> bool:
        """Uses OpenCode to decide if a proposal should be approved."""
        title = proposal.get("title", "")
        desc = proposal.get("description_md", "")
        visible_pixel_hash = proposal.get("visible_pixel_hash", "")
        
        # DETECT RECURSIVE/PROXY PROPOSALS
        if self._is_recursive_proposal(proposal):
            logger.warning(f"Watcher: Detected recursive/proxy proposal {proposal.get('id')}. Rejecting.")
            return False
        
        prompt = (
            f"Audit this technical proposal.\n"
            f"Title: {title}\n"
            f"Plan: {desc[:2000]}...\n\n"
            f"CRITERIA:\n"
            f"1. Must have structured tasks using '### Task X: Title' headers.\n"
            f"2. Must have a technical, non-conversational plan.\n"
            f"3. Must be relevant to the title.\n"
            f"4. Must not be a recursive proposal that just creates another proposal.\n\n"
            f"INSTRUCTION:\n"
            f"Analyze the plan and decide if it meets the criteria. "
            f"Respond with a single line: 'VERDICT: PASS' or 'VERDICT: FAIL - <reason>'."
        )
        
        proposal_id = proposal.get("id", "")
        
        # Try MCP client first for efficiency
        if self.opencode_client.is_available():
            try:
                output = self.opencode_client.run(prompt, timeout=300)  # 5 minutes for proposal audit (quick decision)
                if output:
                    output = output.strip().upper()
                    if "VERDICT: PASS" in output or "PASS" in output.split()[:10]:
                        return True
                    # Cache the rejection reason
                    if proposal_id:
                        self.rejection_cache[proposal_id] = output
                    logger.info(f"Audit failed for {proposal_id}. Auditor output: {output}")
                    return False
            except Exception as e:
                logger.error(f"OpenCode MCP audit failed: {e}")
        
        # Fallback to subprocess if available
        if self.opencode_path:
            import subprocess
            try:
                cmd = ["opencode", "run", prompt]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minutes for proposal audit (quick decision)
                
                if result.returncode != 0:
                    logger.error(f"Auditor: OpenCode failed (exit {result.returncode}): {result.stderr}")
                    return False

                output = result.stdout.strip().upper()
                if "VERDICT: PASS" in output or "PASS" in output.split()[:10]:
                    return True
                    
                # Cache the rejection reason
                if proposal_id:
                    self.rejection_cache[proposal_id] = output
                logger.info(f"Audit failed for {proposal_id}. Auditor output: {output}")
                return False
            except subprocess.TimeoutExpired:
                logger.error(f"Auditor: OpenCode timed out after 300s for proposal {proposal_id}")
                if proposal_id:
                    self.rejection_cache[proposal_id] = "Audit timed out after 300s"
                return False
            except Exception as e:
                logger.error(f"Error during proposal audit: {e}")
                if proposal_id:
                    self.rejection_cache[proposal_id] = f"Audit error: {str(e)}"
                return False
        
        # Auto-approve if no auditor available
        return True

    def process_submissions(self):
        """Audits and reviews (approves/rejects) Worker submissions."""
        submissions = self.client.get_submissions()
        if submissions is None:
            return

        for sub in submissions:
            if not isinstance(sub, dict):
                logger.warning(f"Watcher: process_submissions received non-dict submission: {sub}")
                continue
            sub_id = sub.get("submission_id") or sub.get("id")
            status = sub.get("status", "").lower()
            
            # We look for 'pending_review' or 'submitted'
            if status in ["pending_review", "submitted"] and sub_id not in self.seen_submissions:
                logger.info(f"Watcher auditing submission {sub_id}...")
                
                audit_passed, reason = self._audit_submission(sub)
                
                if audit_passed:
                    logger.info(f"Submission {sub_id} passed audit. Approving...")
                    if sub_id and self.client.review_submission(sub_id, "approve"):
                        logger.info(f"Submission {sub_id} approved.")
                        self.seen_submissions.add(sub_id)
                else:
                    # audit_passed is False, 'reason' contains the Auditor's critique
                    logger.warning(f"Submission {sub_id} failed audit: {reason}")
                    if sub_id and self.client.review_submission(sub_id, "reject", reason):
                        logger.info(f"Submission {sub_id} rejected with feedback.")
                        self.seen_submissions.add(sub_id)

    def _audit_submission(self, sub: Dict) -> Tuple[bool, str]:
        """Uses OpenCode to verify the quality of a submission."""
        deliverables = sub.get("deliverables", {})
        notes = deliverables.get("notes", "")
        
        prompt = (
            f"Audit this work submission report.\n"
            f"Report: {notes[:2000]}\n\n"
            f"CRITERIA:\n"
            f"1. Must contain technical evidence (code, logs, pseudo-code).\n"
            f"2. Must NOT be generic or conversational filler.\n"
            f"3. Must demonstrate task completion.\n\n"
            f"Respond with a single line: 'VERDICT: PASS' or 'VERDICT: FAIL - <reason>'."
        )
        
        # Try MCP client first for efficiency
        if self.opencode_client.is_available():
            try:
                output = self.opencode_client.run(prompt, timeout=300)  # 5 minutes for submission audit (quick decision)
                if output:
                    output = output.strip()
                    if "VERDICT: PASS" in output.upper() or "PASS" in output.upper().split()[:5]:
                        return True, ""
                    return False, output
            except Exception as e:
                logger.error(f"OpenCode MCP submission audit failed: {e}")
        
        # Fallback to subprocess if available
        if self.opencode_path:
            import subprocess
            try:
                cmd = ["opencode", "run", prompt]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minutes for submission audit (quick decision)
                
                if result.returncode != 0:
                    logger.error(f"Auditor: OpenCode failed (exit {result.returncode}): {result.stderr}")
                    return False, "Audit tool failed"

                output = result.stdout.strip()
                if "VERDICT: PASS" in output.upper() or "PASS" in output.upper().split()[:5]:
                    return True, ""
                
                return False, output
            except subprocess.TimeoutExpired:
                logger.error(f"Auditor: OpenCode timed out after 300s for submission {sub.get('id')}")
                return False, "Audit timed out"
            except Exception as e:
                logger.error(f"Error during submission audit: {e}")
                return False, f"Audit error: {str(e)}"
        
        # Auto-approve if no auditor available
        return True, ""

    def _load_state(self):
        """Load persisted state from local file storage."""
        try:
            import os
            import json
            # Use current working directory for sandbox compatibility
            state_file = f"{self.storage_key}.json"
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                if isinstance(state, dict):
                    self.seen_tasks = set(state.get("seen_tasks", []))
                    self.seen_proposals = set(state.get("seen_proposals", []))
                    self.seen_submissions = set(state.get("seen_submissions", []))
                    self.rejected_proposals = set(state.get("rejected_proposals", []))
                    self.rejection_cache = state.get("rejection_cache", {})
                    logger.info(f"Watcher loaded persisted state from file with {len(self.seen_tasks)} tasks, {len(self.rejected_proposals)} rejected proposals")
        except Exception as e:
            logger.warning(f"Failed to load persisted state from file: {e}")

    def _save_state(self):
        """Save current state to local file storage."""
        state_data = {
            "seen_tasks": list(self.seen_tasks),
            "seen_proposals": list(self.seen_proposals), 
            "seen_submissions": list(self.seen_submissions),
            "rejected_proposals": list(self.rejected_proposals),
            "rejection_cache": self.rejection_cache,
            "last_updated": time.time()
        }
        
        # Local file storage only (MCP not available)
        try:
            import os
            import json
            state_file = f"{self.storage_key}.json"
            with open(state_file, 'w') as f:
                json.dump(state_data, f)
            logger.debug("Watcher state saved to local file")
        except Exception as e:
            logger.warning(f"Failed to save persisted state to file: {e}")

    def _check_disk_space(self):
        """Monitor disk usage and cleanup if needed."""
        try:
            import shutil
            import os
            
            # Try multiple common paths for sandbox environments
            paths_to_check = ["/app", ".", "/tmp", os.getcwd()]
            disk_path = None
            
            for path in paths_to_check:
                try:
                    shutil.disk_usage(path)
                    disk_path = path
                    break
                except:
                    continue
            
            if not disk_path:
                logger.warning("Could not find accessible path for disk monitoring")
                return
                
            total, used, free = shutil.disk_usage(disk_path)
            free_gb = free // (1024**3)
            
            # Clean up if less than 5GB free
            if free_gb < 5:
                logger.warning(f"Low disk space: {free_gb}GB free. Running cleanup...")
                self._cleanup_files()
            
            # Log disk usage periodically
            if time.time() - self.last_cleanup_time > self.cleanup_interval:
                used_pct = (used / total) * 100
                logger.info(f"Disk usage at {disk_path}: {used_pct:.1f}% ({free_gb}GB free)")
                self.last_cleanup_time = time.time()
                
        except Exception as e:
            logger.error(f"Disk space check failed: {e}")

    def _cleanup_files(self):
        """Clean up temporary files and old results."""
        try:
            import os
            import glob
            from .config import Config
            
            # Clean old task results (keep only recent 100)
            try:
                results_dir = os.path.join(Config.UPLOADS_DIR, "results")
            except:
                # Fallback to current directory if config fails
                results_dir = "results"
                
            if os.path.exists(results_dir):
                for root, dirs, files in os.walk(results_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            # Remove files older than 24 hours
                            file_age = time.time() - os.path.getmtime(file_path)
                            if file_age > 86400:  # 24 hours
                                os.remove(file_path)
                                logger.debug(f"Cleaned old file: {file_path}")
                        except Exception:
                            pass
                            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def run_once(self) -> List[Dict]:
        try:
            # Check system resources before processing
            self._check_disk_space()
            
            self.process_pending_proposals()
            self.process_submissions()
            
            # Save state periodically
            self._save_state()
            
            return self.find_available_tasks()
        except Exception as e:
            logger.error(f"Watcher encountered error: {e}")
            return []
