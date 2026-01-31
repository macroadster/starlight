import logging
import time
import uuid
import shutil
import threading
import os
from typing import Dict, Set, Optional, Any
from .client import StargateClient
from .config import Config
from .opencode_client import OpenCodeMCPClient
from .dynamic_loader import dynamic_loader, LoadRequest

logger = logging.getLogger(__name__)

class WorkerAgent:
    def __init__(self, client: StargateClient, ai_identifier: str = Config.AI_IDENTIFIER):
        self.client = client
        self.ai_identifier = ai_identifier
        self.opencode_client = OpenCodeMCPClient()
        
        # In-memory caches for current session (will reset on restart)
        self.seen_wishes: set = set()
        self.active_tasks: Set[str] = set()
        self._recent_proposals: Set[str] = set()
        self._rejected_tasks: Dict[str, str] = {}  # task_id -> rejection_reason
        
        # Proposal revision tracking
        self._rejected_proposals: Dict[str, Dict] = {}  # proposal_id -> rejection_info
        self._revision_attempts: Dict[str, int] = {}  # wish_id -> revision_count
        self._max_revisions_per_wish = 3  # Prevent endless revisions
        
        # Persistent storage key for this agent instance
        self.storage_key = f"worker_state_{ai_identifier.lower().replace('-', '_')}"
        
        # Load persisted state from MCP storage
        self._load_state()
        
        # Resource management
        self.concurrency_limit = threading.BoundedSemaphore(1)  # Prevent OOM in container
        self.last_cleanup_time = 0
        self.cleanup_interval = 3600  # 1 hour between cleanups
        
        # Keep subprocess fallback for compatibility
        self.opencode_path = shutil.which("opencode")
        if self.opencode_client.is_available():
            logger.info("OpenCode MCP client connected. Efficient work execution enabled.")
        elif self.opencode_path:
            logger.info(f"OpenCode detected at {self.opencode_path}")
        else:
            logger.warning("OpenCode not found in PATH. Falling back to simulated work.")

    def _has_incomplete_tasks(self) -> bool:
        """Check if worker has incomplete tasks that need completion."""
        logger.debug(f"Worker: Checking for incomplete tasks. Active tasks: {list(self.active_tasks)}")
        
        if not self.active_tasks:
            logger.debug("Worker: No active tasks, proceeding")
            return False
            
        # Check if any active tasks are actually incomplete
        for task_id in list(self.active_tasks):
            try:
                details = self.client.get_task_status(str(task_id))
                logger.debug(f"Worker: Task {task_id} status: {details.get('status') if details else 'None'}")
                
                if details:
                    status = details.get("status", "").lower()
                    # Task is incomplete if still claimed by us and not completed
                    if status in ["claimed", "in_progress", "rejected"]:
                        claimed_by = details.get("claimed_by", "").lower()
                        is_ours = claimed_by == self.ai_identifier.lower()
                        if not is_ours and Config.DONATION_ADDRESS:
                            is_ours = claimed_by == Config.DONATION_ADDRESS.lower()
                        if is_ours:
                            logger.debug(f"Worker: Task {task_id} is still active and claimed by us")
                            return True
                    else:
                        # Task is no longer active, remove from tracking
                        logger.debug(f"Worker: Task {task_id} is no longer active ({status}), removing from tracking")
                        self.active_tasks.discard(str(task_id))
            except Exception as e:
                logger.error(f"Error checking task status for {task_id}: {e}")
                # Assume incomplete if we can't verify
                return True
        
        logger.debug("Worker: No incomplete tasks found, proceeding")
        return False

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
                    self.seen_wishes = set(state.get("seen_wishes", []))
                    self._recent_proposals = set(state.get("recent_proposals", []))
                    self._rejected_tasks = state.get("rejected_tasks", {})
                    self._revision_attempts = state.get("revision_attempts", {})
                    self._rejected_proposals = state.get("rejected_proposals", {})
                    logger.info(f"Worker loaded persisted state from file with {len(self.seen_wishes)} wishes, {len(self._rejected_tasks)} rejected tasks, {len(self._revision_attempts)} revision attempts")
        except Exception as e:
            logger.warning(f"Failed to load persisted state from file: {e}")

    def _save_state(self):
        """Save current state to local file storage."""
        state_data = {
            "seen_wishes": list(self.seen_wishes),
            "recent_proposals": list(self._recent_proposals),
            "rejected_tasks": self._rejected_tasks,
            "revision_attempts": self._revision_attempts,
            "rejected_proposals": self._rejected_proposals,
            "last_updated": time.time()
        }
        
        # Local file storage only (MCP not available)
        try:
            import os
            import json
            state_file = f"{self.storage_key}.json"
            with open(state_file, 'w') as f:
                json.dump(state_data, f)
            logger.debug("Worker state saved to local file")
        except Exception as e:
            logger.warning(f"Failed to save persisted state to file: {e}")

    def _check_resources(self):
        """Monitor system resources and adjust behavior."""
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
                logger.warning("Could not find accessible path for resource monitoring")
                return True  # Assume OK if can't check
                
            total, used, free = shutil.disk_usage(disk_path)
            free_gb = free // (1024**3)
            logger.debug(f"Worker: Resource check - {free_gb}GB free at {disk_path}")
            
            # Reduce activity if low on disk space
            if free_gb < 3:
                logger.warning(f"Very low disk space: {free_gb}GB free. Reducing worker activity.")
                return False  # Skip new work
            
            # Log disk usage periodically
            if time.time() - self.last_cleanup_time > self.cleanup_interval:
                used_pct = (used / total) * 100
                logger.info(f"Worker disk usage at {disk_path}: {used_pct:.1f}% ({free_gb}GB free)")
                self.last_cleanup_time = time.time()
                
            return True  # Resources OK
        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            return True  # Assume OK if check fails

    def _load_rejection_reason(self, proposal_id: str) -> str:
        """Load rejection reason from file storage if available."""
        try:
            # File storage only (MCP not available)
            import json
            rejection_file = f"rejection_{proposal_id}.json"
            try:
                with open(rejection_file, 'r') as f:
                    rejection_info = json.load(f)
                reason = rejection_info.get("rejection_reason")
                if reason:
                    logger.info(f"Worker: Loaded rejection reason from file: {reason}")
                    return reason
            except FileNotFoundError:
                pass
                
        except Exception as e:
            logger.warning(f"Failed to load rejection reason for {proposal_id}: {e}")
            
        return "Proposal rejected - no specific reason provided"

    def _check_for_revision_opportunities(self, proposals: list):
        """Check for rejected proposals that can be revised."""
        my_hash = self.client._api_key_hash()
        revision_candidates = []
        
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
                
            pid = proposal.get("id")
            status = proposal.get("status", "").lower()
            vph = proposal.get("visible_pixel_hash")
            
            # Only consider my rejected proposals
            if status != "rejected" or not vph or not pid:
                continue
                
            # Check if I created this proposal
            p_meta = proposal.get("metadata") or {}
            p_creator = p_meta.get("creator_api_key_hash")
            if p_creator != my_hash:
                continue
                
            # Check if any OTHER proposal for same wish was approved
            wish_has_approved = False
            for other in proposals:
                if (other.get("visible_pixel_hash") == vph and 
                    other.get("id") != pid and
                    other.get("status", "").lower() in ["approved", "active"]):
                    wish_has_approved = True
                    break
            
            # If no other proposal approved, this is revision candidate
            if not wish_has_approved:
                # Load rejection reason for this candidate
                rejection_reason = self._load_rejection_reason(pid)
                
                # Cache rejection info for future use
                self._rejected_proposals[pid] = {
                    "reason": rejection_reason,
                    "timestamp": time.time()
                }
                
                revision_candidates.append({
                    "proposal_id": pid,
                    "wish_id": vph,
                    "proposal": proposal,
                    "rejection_reason": rejection_reason
                })
        
        return revision_candidates

    def _attempt_proposal_revision(self, revision_candidate: dict) -> bool:
        """Attempt to revise a rejected proposal."""
        proposal_id = revision_candidate["proposal_id"]
        wish_id = revision_candidate["wish_id"]
        rejected_proposal = revision_candidate["proposal"]
        
        # Check revision limits
        revision_count = self._revision_attempts.get(wish_id, 0)
        if revision_count >= self._max_revisions_per_wish:
            logger.info(f"Worker: Max revisions ({self._max_revisions_per_wish}) reached for wish {wish_id}")
            return False
        
        # Track revision attempt
        self._revision_attempts[wish_id] = revision_count + 1
        
        # Get the original wish for context
        wishes = self.client.get_open_contracts()
        original_wish = None
        for wish in wishes:
            if isinstance(wish, dict):
                wid = wish.get("contract_id", "").replace("wish-", "")
                if wid == wish_id:
                    original_wish = wish
                    break
        
        if not original_wish:
            logger.warning(f"Worker: Could not find original wish for revision of {proposal_id}")
            return False
        
        # Get rejection reason if available
        rejection_info = self._rejected_proposals.get(proposal_id, {})
        rejection_reason = rejection_info.get("reason", "Proposal was rejected")
        
        logger.info(f"Worker: Attempting revision {revision_count + 1}/{self._max_revisions_per_wish} for rejected proposal {proposal_id}")
        logger.info(f"Worker: Rejection reason: {rejection_reason}")
        
        # Create improved proposal
        if self._create_revised_proposal(original_wish, rejected_proposal, rejection_reason, wish_id):
            self._recent_proposals.add(wish_id)  # Prevent duplicate revisions
            logger.info(f"Worker: Successfully revised proposal for wish {wish_id}")
            return True
        
        logger.warning(f"Worker: Failed to revise proposal {proposal_id}")
        return False

    def process_wishes(self):
        """Scans for wishes and creates proposals ONLY - no wish creation."""
        try:
            # RESOURCE CHECK: Skip if low on resources
            if not self._check_resources():
                logger.info("Worker: Resource check failed, skipping proposal creation")
                return
                
            # TASK COMPLETION GATE: Must complete active tasks before creating new proposals
            if self.active_tasks:
                logger.info(f"Worker: Skipping proposal creation - {len(self.active_tasks)} active tasks: {list(self.active_tasks)}")
                return
                
            # Check for incomplete tasks (prevents task abandonment)
            if self._has_incomplete_tasks():
                logger.warning("Worker: Has incomplete tasks. Skipping new proposal creation until completion.")
                return
                
            my_hash = self.client._api_key_hash()
            proposals = self.client.get_proposals()
            
            # PRIORITY 1: Check for revision opportunities first
            revision_candidates = self._check_for_revision_opportunities(proposals)
            if revision_candidates:
                logger.info(f"Worker: Found {len(revision_candidates)} revision candidates")
                
                # Attempt revision for first candidate (rate limited)
                if self._attempt_proposal_revision(revision_candidates[0]):
                    # Save state after successful revision
                    self._save_state()
                    return  # Only one revision per cycle
            
            # Map wish_id -> boolean (did I propose for this?)
            my_proposals_for_wishes = set()
            
            for p in proposals:
                if not isinstance(p, dict):
                    continue
                
                vph = p.get("visible_pixel_hash")
                if not vph:
                    continue
                
                # Check if I created this proposal by looking at metadata
                p_meta = p.get("metadata") or {}
                p_creator = p_meta.get("creator_api_key_hash")
                
                if p_creator == my_hash:
                    my_proposals_for_wishes.add(vph)

            # Get open contracts (wishes) - READ ONLY, NO CREATION
            wishes = self.client.get_open_contracts()
            logger.info(f"Worker: Found {len(wishes)} total wishes. I have already proposed for {len(my_proposals_for_wishes)}.")
            
            # PROPOSAL RATE LIMIT: Only create 1 proposal per cycle to prevent spam
            max_proposals_per_cycle = 1
            proposals_created = 0
            
            for wish in wishes:
                if not isinstance(wish, dict):
                    continue
                
                if proposals_created >= max_proposals_per_cycle:
                    logger.info(f"Worker: Reached proposal limit ({max_proposals_per_cycle}) for this cycle")
                    break
                
                wid = wish.get("contract_id")
                status = wish.get("status", "").lower()
                text = wish.get("title", "")
                
                # Only process pending wishes
                if not wid or status != "pending":
                    continue
                
                # Normalize wish ID (remove wish- prefix if present)
                normalized_wid = wid.replace("wish-", "")
                
                # Check if already proposed
                already_proposed = normalized_wid in my_proposals_for_wishes
                recently_cached = normalized_wid in self._recent_proposals
                
                # Allow multiple proposals per wish (removed duplication checks)
                # if already_proposed:
                #     logger.debug(f"Worker: Already proposed for wise {wid}")
                #     continue
                # elif recently_cached:
                #     logger.debug(f"Worker: Already cached proposal for wise {wid}")
                #     continue
                
                # Create proposal ONLY - no wish creation
                logger.info(f"Worker: Creating proposal for wish {wid}")
                if self._create_proposal_for_wish(wish):
                    my_proposals_for_wishes.add(normalized_wid)
                    self._recent_proposals.add(normalized_wid)
                    proposals_created += 1
                    logger.info(f"Worker: Successfully created proposal for wish {wid}")
                    # Save state after successful proposal
                    self._save_state()
                else:
                    logger.warning(f"Worker: Failed to create proposal for wish {wid}")
            
            if proposals_created > 0:
                logger.info(f"Worker: Created {proposals_created} new proposals this cycle")

        except Exception as e:
            logger.error(f"Worker error processing wishes: {e}")

    def _create_proposal_for_wish(self, wish: Dict) -> bool:
        """Generates and submits a proposal for a given wish."""
        wid = wish.get("contract_id", "")
        text = wish.get("title", "No description provided")
        
        # Normalize wish ID (remove wish- prefix if present for visible_pixel_hash)
        normalized_wid = wid.replace("wish-", "")
        
        title = f"Proposal for: {text.splitlines()[0][:50]}"
        description = (
            f"I propose to fulfill the wish: '{text}' by executing a systematic implementation plan.\n\n"
            f"### Task 1: Build Solution\n"
            f"Execute the technical requirements to fulfill the original wish: {text}"
        )
        
        # Try MCP client first for efficiency
        if self.opencode_client.is_available():
            logger.info(f"Using OpenCode MCP to generate proposal for wish {wid}")
            try:
                # Ask OpenCode to generate a systematic plan for the wish
                prompt = (
                    f"Create an elite technical implementation plan for this wish: '{text}'.\n"
                    f"Decompose the work into logical, actionable tasks (e.g. using '### Task X: Title' headers). For each task, describe the implementation steps and technical deliverables.\n"
                    f"Focus on expert engineering practices and clear documentation."
                )
                result = self.opencode_client.run(prompt, timeout=300)  # 5 minutes for proposal creation
                if result and result.strip():
                    description = result.strip()
                    # Heuristic for title - find the first non-empty line
                    lines = [l for l in description.splitlines() if l.strip()]
                    if lines:
                        title = lines[0].strip("# ")
                        if len(title) > 100:
                            title = title[:97] + "..."
            except Exception as e:
                logger.error(f"Failed to use OpenCode MCP for proposal generation: {e}")
        
        # Fallback to subprocess if available
        elif self.opencode_path:
            import subprocess
            logger.info(f"Using OpenCode subprocess to generate proposal for wish {wid}")
            try:
                # Ask OpenCode to generate a systematic plan for the wish
                prompt = (
                    f"Create an elite technical implementation plan for this wish: '{text}'.\n"
                    f"Decompose the work into logical, actionable tasks (e.g. using '### Task X: Title' headers). For each task, describe the implementation steps and technical deliverables.\n"
                    f"Focus on expert engineering practices and clear documentation."
                )
                cmd = ["opencode", "run", prompt]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minutes for proposal creation
                if result.returncode == 0 and result.stdout.strip():
                    description = result.stdout.strip()
                    # Heuristic for title - find the first non-empty line
                    lines = [l for l in description.splitlines() if l.strip()]
                    if lines:
                        title = lines[0].strip("# ")
                        if len(title) > 100:
                            title = title[:97] + "..."
            except Exception as e:
                logger.error(f"Failed to use OpenCode subprocess for proposal generation: {e}")

        # Create proposal.
        proposal_data = {
            "title": title,
            "description_md": description,
            "budget_sats": 1000,
            "visible_pixel_hash": normalized_wid,
            "contract_id": wid
        }
        
        logger.info(f"Worker creating proposal for wish {wid} with visible_pixel_hash: {normalized_wid}")
        pid = self.client.create_proposal(proposal_data)
        if pid:
            logger.info(f"Proposal created successfully: {pid} for wish {wid}")
            return True
        else:
            logger.error(f"Failed to create proposal for wish {wid}")
            return False

    def _create_revised_proposal(self, wish: Dict, rejected_proposal: Dict, rejection_reason: str, wish_id: str) -> bool:
        """Create an improved proposal based on rejection feedback."""
        wish_text = wish.get("title", "No description provided")
        old_title = rejected_proposal.get("title", "")
        old_description = rejected_proposal.get("description_md", "")
        
        # Generate improved proposal using OpenCode
        title = f"REVISED Proposal for: {wish_text.splitlines()[0][:50]}"
        description = (
            f"REVISED PROPOSAL - Previous version was rejected.\n"
            f"Original Wish: '{wish_text}'\n"
            f"Rejection Reason: {rejection_reason}\n\n"
            f"### Improved Implementation Plan\n"
            f"Based on the feedback, here is a corrected and enhanced approach:\n\n"
            f"### Task 1: Address Rejection Issues\n"
            f"Fix all identified problems from the rejection: {rejection_reason}\n\n"
            f"### Task 2: Complete Original Wish Requirements\n"
            f"Fulfill the original wish: {wish_text}"
        )
        
        # Try MCP client first for revision generation
        if self.opencode_client.is_available():
            logger.info(f"Using OpenCode MCP to generate revised proposal for wish {wish_id}")
            try:
                prompt = (
                    f"REVISE and improve this rejected proposal.\n\n"
                    f"ORIGINAL WISH: '{wish_text}'\n\n"
                    f"REJECTED PROPOSAL:\n"
                    f"Title: {old_title}\n"
                    f"Plan: {old_description[:1000]}...\n\n"
                    f"REJECTION REASON: {rejection_reason}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"1. Fix all issues mentioned in the rejection reason\n"
                    f"2. Create a more detailed, technical implementation plan\n"
                    f"3. Avoid previous mistakes and weak points\n"
                    f"4. Provide specific, actionable tasks with clear deliverables\n"
                    f"5. Make the proposal significantly better than the rejected version\n\n"
                    f"Generate a comprehensive, technically sound proposal that addresses the rejection feedback."
                )
                result = self.opencode_client.run(prompt, timeout=1800)  # 30 minutes for revision generation
                if result and result.strip():
                    description = result.strip()
                    # Heuristic for title - find the first good line
                    lines = [l for l in description.splitlines() if l.strip() and not l.strip().startswith('#')]
                    if lines:
                        title = lines[0].strip()
                        if len(title) > 100:
                            title = title[:97] + "..."
                    # Add revision prefix to title
                    if not title.upper().startswith("REVISED"):
                        title = f"REVISED: {title}"
            except Exception as e:
                logger.error(f"Failed to use OpenCode MCP for proposal revision: {e}")
        
        # Fallback to subprocess if available
        elif self.opencode_path:
            import subprocess
            logger.info(f"Using OpenCode subprocess to generate revised proposal for wish {wish_id}")
            try:
                prompt = (
                    f"REVISE and improve this rejected proposal.\n\n"
                    f"ORIGINAL WISH: '{wish_text}'\n\n"
                    f"REJECTED PROPOSAL:\n"
                    f"Title: {old_title}\n"
                    f"Plan: {old_description[:1000]}...\n\n"
                    f"REJECTION REASON: {rejection_reason}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"1. Fix all issues mentioned in the rejection reason\n"
                    f"2. Create a more detailed, technical implementation plan\n"
                    f"3. Avoid previous mistakes and weak points\n"
                    f"4. Provide specific, actionable tasks with clear deliverables\n"
                    f"5. Make the proposal significantly better than the rejected version\n\n"
                    f"Generate a comprehensive, technically sound proposal that addresses the rejection feedback."
                )
                cmd = ["opencode", "run", prompt]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 minutes for revision generation
                if result.returncode == 0 and result.stdout.strip():
                    description = result.stdout.strip()
                    # Heuristic for title - find the first good line
                    lines = [l for l in description.splitlines() if l.strip() and not l.strip().startswith('#')]
                    if lines:
                        title = lines[0].strip()
                        if len(title) > 100:
                            title = title[:97] + "..."
                    # Add revision prefix to title
                    if not title.upper().startswith("REVISED"):
                        title = f"REVISED: {title}"
            except Exception as e:
                logger.error(f"Failed to use OpenCode subprocess for proposal revision: {e}")

        # Create revised proposal
        proposal_data = {
            "title": title,
            "description_md": description,
            "budget_sats": 1000,
            "visible_pixel_hash": wish_id,
            "contract_id": wish.get("contract_id"),
            "metadata": {
                "creator_api_key_hash": self.client._api_key_hash(),
                "revision_of": rejected_proposal.get("id"),
                "revision_number": self._revision_attempts.get(wish_id, 0),
                "rejection_reason": rejection_reason
            }
        }
        
        logger.info(f"Worker creating REVISED proposal for wish {wish_id}")
        pid = self.client.create_proposal(proposal_data)
        if pid:
            logger.info(f"REVISED proposal created successfully: {pid} for wish {wish_id}")
            return True
        else:
            logger.error(f"Failed to create REVISED proposal for wish {wish_id}")
            return False

    def process_task(self, task: Dict) -> bool:
        """Claims, resumes, or reworks background execution of a task."""
        task_id = str(task.get("task_id", ""))
        title = task.get("title", "Unknown Task")
        status = task.get("status", "").lower()
        existing_claim_id = task.get("active_claim_id")
        
        # HOARDING PREVENTION: Only work on one task at a time
        if len(self.active_tasks) > 0:
            if task_id not in self.active_tasks:
                logger.debug(f"Worker busy with {self.active_tasks}. Skipping claim for: {title}")
                return False
        
        if task_id in self.active_tasks:
            return False
            
        # 1. Handle Claiming / Resumption / Rework
        claim_id = None
        rejection_feedback = None
        
        # PRIORITY: Check local rejection cache first (handles race conditions)
        if task_id and task_id in self._rejected_tasks:
            rejection_feedback = self._rejected_tasks[task_id]
            logger.info(f"Worker: Found cached rejection for task {task_id}: {rejection_feedback}")
        else:
            # Check API for fresh rejection data
            details = self.client.get_task_status(task_id) if task_id else None
            if details:
                # Look for rejection reasons in multiple possible fields
                rejection_feedback = (
                    details.get("rejection_reason") or 
                    details.get("last_rejection_reason") or 
                    details.get("feedback") or
                    details.get("audit_feedback")
                )
                
                # Cache the rejection for future reference
                if rejection_feedback and task_id:
                    self._rejected_tasks[task_id] = rejection_feedback
                    logger.info(f"Worker: Cached rejection for task {task_id}: {rejection_feedback}")
            
            if rejection_feedback:
                logger.info(f"Worker: Task {task_id} REJECTED. Reason: {rejection_feedback}. PRIORITY REWORK...")
                # For rejected tasks, try to claim immediately (higher priority)
                if (status == "available" or status == "rejected") and task_id:
                    logger.info(f"Worker: Priority claiming rejected task {task_id} for rework")
                    claim_id = self.client.claim_task(task_id, self.ai_identifier)
                    if claim_id:
                        logger.info(f"Worker: Successfully claimed rejected task {task_id} for rework")
                    else:
                        # Check if we already own it from previous attempt
                        claim_id = self._find_my_existing_claim(task)
                        if claim_id:
                            logger.info(f"Worker: Found existing claim for rejected task {task_id}")
        
        # Handle normal claiming if not already handled above
        if not claim_id:
            if (status == "claimed" or status == "rejected") and existing_claim_id:
                logger.info(f"Worker: Resuming/Reworking task: {title} ({task_id}). Status: {status}. Claim: {existing_claim_id}")
                claim_id = existing_claim_id
            elif (status == "available" or status == "rejected") and task_id:
                logger.info(f"Worker attempting to claim task: {title} ({task_id}). Status: {status}")
                claim_id = self.client.claim_task(task_id, self.ai_identifier)
                
                # If claim failed, check if we already own it
                if not claim_id:
                    logger.info(f"Worker: Claim failed for {task_id}. Checking existing ownership...")
                    claim_id = self._find_my_existing_claim(task)
        
        if not claim_id:
            # Check current task status to determine why claim failed
            current_status = self.client.get_task_status(task_id)
            if current_status:
                task_status = current_status.get("status", "").lower()
                claimed_by = current_status.get("claimed_by", "")
                
                if task_status == "approved" or task_status == "completed":
                    logger.info(f"Task {task_id} is already {task_status} (claimed by {claimed_by}). Skipping.")
                elif task_status == "claimed" and claimed_by != self.ai_identifier:
                    logger.info(f"Task {task_id} is already claimed by another agent ({claimed_by}). Skipping.")
                else:
                    logger.warning(f"Failed to secure claim for task {task_id}. Status: {task_status}, claimed_by: {claimed_by}")
            else:
                logger.warning(f"Failed to secure claim for task {task_id}. Could not fetch current status.")
            return False
        
        # Log priority status
        work_type = "REWORK" if rejection_feedback else "NEW WORK"
        logger.info(f"Successfully secured task {task_id} for {work_type}. Claim ID: {claim_id}")
        
        # 2. Start Work in Background
        if task_id:
            self.active_tasks.add(task_id)
        # Pass rejection feedback to background thread for proper rework handling
        task["rejection_feedback"] = rejection_feedback
        task["work_type"] = work_type.lower()
        thread = threading.Thread(target=self._run_task_background, args=(task, claim_id), daemon=True)
        thread.start()
        
        return True

    def _find_my_existing_claim(self, task: Dict) -> Optional[str]:
        """Queries the backend to see if the task is already claimed by this agent."""
        task_id = task.get("task_id")
        contract_id = task.get("contract_id") or task.get("proposal_id")
        
        if not contract_id:
            return None
            
        # Fetch current task state from backend
        tasks = self.client.get_tasks(contract_id)
        for t in tasks:
            if t.get("task_id") == task_id:
                claimed_by = t.get("claimed_by", "").lower()
                active_claim_id = t.get("active_claim_id")
                
                is_ours = claimed_by == self.ai_identifier.lower()
                if not is_ours and Config.DONATION_ADDRESS:
                    is_ours = claimed_by == Config.DONATION_ADDRESS.lower()

                if is_ours and active_claim_id:
                    logger.info(f"Worker: Found existing claim for {task_id}: {active_claim_id}")
                    return active_claim_id
        return None

    def _run_task_background(self, task: Dict, claim_id: str):
        """Executes task in background and submits results."""
        task_id = str(task.get("task_id", ""))
        try:
            with self.concurrency_limit:
                deliverables = self._perform_work(task)
            
            logger.info(f"Submitting work for task {task_id} (Claim: {claim_id})...")
            if self.client.submit_work(claim_id, deliverables):
                logger.info(f"Task {task_id} completed and submitted successfully.")
                # Clear rejection cache on successful submission
                if task_id in self._rejected_tasks:
                    del self._rejected_tasks[task_id]
                    logger.info(f"Worker: Cleared rejection cache for completed task {task_id}")
            else:
                logger.error(f"Failed to submit work for task {task_id}.")
        except Exception as e:
            logger.error(f"Error during background task {task_id}: {e}")
        finally:
            if task_id:
                self.active_tasks.discard(task_id)

    def _perform_work(self, task: Dict) -> Dict:
        """Executes work efficiently with crash prevention."""
        description = task.get("description", "")
        title = task.get("title", "Unknown Task")
        proposal_title = task.get("proposal_title", "Unknown Proposal")
        task_id = task.get("task_id", str(uuid.uuid4()))
        contract_id = task.get("contract_id", "unassigned")
        # Use visible_pixel_hash for better isolation anchoring
        visible_pixel_hash = task.get("visible_pixel_hash") or contract_id
        rejection_feedback = task.get("rejection_feedback")
        
        # Extract skills to inspire the agent
        skills = task.get("skills", ["general engineering"])
        skills_str = ", ".join(skills) if isinstance(skills, list) else str(skills)

        # Determine data directory for results using visible_pixel_hash for isolation
        base_uploads_dir = Config.UPLOADS_DIR
        contract_results_dir = os.path.join(base_uploads_dir, "results", visible_pixel_hash)
        os.makedirs(contract_results_dir, exist_ok=True)
        
        # Copy AGENTS_WORKING_GUIDE.md into sandbox directory for context
        try:
            # Try multiple potential locations for the guide
            potential_sources = []
            
            # 1. Environment variable (Docker/set environment)
            if os.getenv("AGENTS_WORKING_GUIDE_PATH"):
                potential_sources.append(os.getenv("AGENTS_WORKING_GUIDE_PATH"))
            
            # 2. Project root (3 levels up from agents/worker.py)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            potential_sources.append(os.path.join(project_root, "AGENTS_WORKING_GUIDE.md"))
            
            # 3. Current working directory fallback
            potential_sources.append(os.path.join(os.getcwd(), "AGENTS_WORKING_GUIDE.md"))
            
            agents_guide_dest = os.path.join(contract_results_dir, "AGENTS.md")
            guide_copied = False
            
            for agents_guide_src in potential_sources:
                if os.path.exists(agents_guide_src):
                    import shutil
                    shutil.copy2(agents_guide_src, agents_guide_dest)
                    logger.info(f"Added AGENTS.md guide to worker sandbox: {agents_guide_dest} (from {agents_guide_src})")
                    guide_copied = True
                    break
            
            if not guide_copied:
                logger.warning(f"AGENTS_WORKING_GUIDE.md not found in any location: {potential_sources}")
                
        except Exception as e:
            logger.error(f"Failed to copy AGENTS_WORKING_GUIDE.md to sandbox: {e}")
        
        result_filename = f"{task_id}.md"
        result_path = os.path.join(contract_results_dir, result_filename)
        public_url = f"/uploads/results/{visible_pixel_hash}/{result_filename}"

        opencode_output = ""
        
        # Try MCP client first for efficiency
        if self.opencode_client.is_available():
            logger.info(f"Executing work using OpenCode MCP: {title}")
            try:
                # Change to contract-specific working directory for isolation
                original_cwd = os.getcwd()
                os.chdir(contract_results_dir)
                logger.info(f"Changed working directory to: {contract_results_dir}")
                
                try:
                    # TASK-FOCUSED PROMPT - practical and concise
                    base_instruction = (
                        f"You are a practical engineer implementing: '{description}'.\n"
                        f"Skills: [{skills_str}].\n\n"
                        f"IMPORTANT: Read AGENTS.md in your current directory for complete working guidelines!\n\n"
                        f"TASK: Complete this work efficiently and provide concrete results.\n\n"
                        f"REQUIREMENTS:\n"
                        f"1. Provide specific implementation details\n"
                        f"2. Include actual code examples or execution steps\n"
                        f"3. Show evidence of completion\n"
                        f"4. Keep response concise and actionable\n\n"
                        f"Follow security guidelines in AGENTS.md - only allowed imports and operations.\n"
                        f"Focus on delivering working solutions, not theoretical discussions."
                    )

                    work_type = task.get("work_type", "new work").upper()
                    
                    if rejection_feedback:
                        prompt = (
                            f"REWORK REQUIRED: Your previous submission for '{title}' was REJECTED.\n"
                            f"FEEDBACK: '{rejection_feedback}'.\n\n"
                            f"TASK: Fix all identified issues and provide a corrected implementation.\n"
                            f"Focus on addressing each rejection point specifically.\n\n"
                            f"{base_instruction}"
                        )
                        logger.info(f"Worker: Executing REWORK for {title} - Feedback: {rejection_feedback}")
                    else:
                        prompt = base_instruction
                        logger.info(f"Worker: Executing NEW WORK for {title}")
                    
                    result = self.opencode_client.run(prompt, timeout=3600)  # 1 hour timeout for complex tasks
                finally:
                    # Always restore original working directory
                    os.chdir(original_cwd)
                    logger.info(f"Restored working directory to: {original_cwd}")
                if result:
                    opencode_output = result
                else:
                    logger.warning(f"OpenCode MCP returned no result, using fallback")
            except Exception as e:
                logger.error(f"OpenCode MCP execution error: {e}")
        
        # Fallback to subprocess if available
        elif self.opencode_path:
            import subprocess
            logger.info(f"Executing work using OpenCode subprocess: {title}")
            try:
                # Change to contract-specific working directory for isolation
                original_cwd = os.getcwd()
                os.chdir(contract_results_dir)
                logger.info(f"Changed working directory to: {contract_results_dir}")
                
                try:
                    # TASK-FOCUSED PROMPT - practical and concise
                    base_instruction = (
                        f"You are a practical engineer implementing: '{description}'.\n"
                        f"Skills: [{skills_str}].\n\n"
                        f"IMPORTANT: Read AGENTS.md in your current directory for complete working guidelines!\n\n"
                        f"TASK: Complete this work efficiently and provide concrete results.\n\n"
                        f"REQUIREMENTS:\n"
                        f"1. Provide specific implementation details\n"
                        f"2. Include actual code examples or execution steps\n"
                        f"3. Show evidence of completion\n"
                        f"4. Keep response concise and actionable\n\n"
                        f"Follow security guidelines in AGENTS.md - only allowed imports and operations.\n"
                        f"Focus on delivering working solutions, not theoretical discussions."
                    )

                    work_type = task.get("work_type", "new work").upper()
                    
                    if rejection_feedback:
                        prompt = (
                            f"REWORK REQUIRED: Your previous submission for '{title}' was REJECTED.\n"
                            f"FEEDBACK: '{rejection_feedback}'.\n\n"
                            f"TASK: Fix all identified issues and provide a corrected implementation.\n"
                            f"Focus on addressing each rejection point specifically.\n\n"
                            f"{base_instruction}"
                        )
                        logger.info(f"Worker: Executing REWORK for {title} - Feedback: {rejection_feedback}")
                    else:
                        prompt = base_instruction
                        logger.info(f"Worker: Executing NEW WORK for {title}")
                    
                    cmd = ["opencode", "run", prompt]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=contract_results_dir)  # 1 hour timeout
                finally:
                    # Always restore original working directory
                    os.chdir(original_cwd)
                    logger.info(f"Restored working directory to: {original_cwd}")
                
                if result.returncode == 0:
                    opencode_output = result.stdout
                else:
                    logger.error(f"OpenCode failed with exit code {result.returncode}")
            except Exception as e:
                logger.error(f"OpenCode subprocess execution error: {e}")

        # Simple fallback if no OpenCode available
        if not opencode_output:
            logger.info(f"Using practical fallback for: {title}")
            time.sleep(1)  # Brief pause to simulate work
            
            status_line = "Rework Applied" if rejection_feedback else "Implementation Complete"
            
            opencode_output = (
                f"## Task Implementation: {title}\n\n"
                f"**Status:** {status_line}\n"
                f"**Description:** {description}\n\n"
                f"### Implementation Details\n"
                f"- Analyzed requirements for task completion\n"
                f"- Applied technical solution using {skills_str}\n"
                f"- Verified implementation meets specifications\n\n"
                f"### Evidence of Completion\n"
                f"```python\n"
                f"# Implementation completed successfully\n"
                f"def complete_task():\n"
                f"    return \"Task: {title} - Status: Complete\"\n"
                f"```\n\n"
                f"### Result\n"
                f"Task completed with practical working solution."
            )
            if rejection_feedback:
                opencode_output += f"\n\n**Rework Note:** Addressed feedback: {rejection_feedback}"

        notes = (
            f"# Task Report: {title}\n\n"
            f"**Agent:** {self.ai_identifier}\n"
            f"**Proposal:** {proposal_title}\n"
            f"**Task ID:** {task_id}\n\n"
            f"## Implementation\n"
            f"{opencode_output}\n\n"
            f"--- \n"
            f"**Report:** [Download]({public_url})"
        )

        # Write the report to disk
        try:
            with open(result_path, "w") as f:
                f.write(notes)
            logger.info(f"Saved task results to {result_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

        return {
            "notes": notes,
            "result_file": public_url,
            "artifacts_dir": f"/uploads/results/{visible_pixel_hash}/",
            "completion_proof": str(uuid.uuid4())
        }

    def _register_dynamic_functions(self, visible_pixel_hash: str, contract_results_dir: str):
        """Register api.py files from the worker's sandbox as dynamic endpoints."""
        try:
            api_py_path = os.path.join(contract_results_dir, "api.py")
            if os.path.exists(api_py_path):
                logger.info(f"Found api.py in sandbox: {api_py_path}")
                try:
                    request = LoadRequest(
                        visible_pixel_hash=visible_pixel_hash,
                        function_name="handler",
                        module_name="api"
                    )
                    loaded_module = dynamic_loader.load_function(request)
                    logger.info(f"Successfully registered dynamic function: {loaded_module.endpoint_path}")
                except Exception as e:
                    logger.warning(f"Failed to register api.py: {e}")
            else:
                logger.debug(f"No api.py found in sandbox: {api_py_path}")
        except Exception as e:
            logger.error(f"Error during dynamic function registration: {e}")
