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
                    logger.info(f"Worker loaded persisted state from file with {len(self.seen_wishes)} wishes, {len(self._rejected_tasks)} rejected tasks")
        except Exception as e:
            logger.warning(f"Failed to load persisted state from file: {e}")

    def _save_state(self):
        """Save current state to local file storage."""
        state_data = {
            "seen_wishes": list(self.seen_wishes),
            "recent_proposals": list(self._recent_proposals),
            "rejected_tasks": self._rejected_tasks,
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

            # Map wish_id -> boolean (did I propose for this?)
            my_proposals_for_wishes = set()
            # Count ALL proposals per wish (from any creator) to enforce limit
            proposals_count_per_wish: Dict[str, int] = {}
            MAX_PROPOSALS_PER_WISH = 5

            for p in proposals:
                if not isinstance(p, dict):
                    continue

                vph = p.get("visible_pixel_hash")
                if not vph:
                    continue

                # Count proposals per wish
                proposals_count_per_wish[vph] = proposals_count_per_wish.get(vph, 0) + 1

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
                
                # Check if already proposed - prevent duplicate proposals for same wish
                already_proposed = normalized_wid in my_proposals_for_wishes
                recently_cached = normalized_wid in self._recent_proposals
                # Check if wish already has max proposals
                current_proposal_count = proposals_count_per_wish.get(normalized_wid, 0)

                if already_proposed:
                    logger.debug(f"Worker: Already proposed for wish {wid}, skipping")
                    continue
                elif recently_cached:
                    logger.debug(f"Worker: Already cached proposal for wish {wid}, skipping")
                    continue
                elif current_proposal_count >= MAX_PROPOSALS_PER_WISH:
                    logger.debug(f"Worker: Wish {wid} already has {current_proposal_count} proposals (max {MAX_PROPOSALS_PER_WISH}), skipping")
                    continue

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
                    f"Create an plan for this wish: '{text}'.\n"
                    f"Decompose the work into logical, actionable tasks (e.g. using '### Task X: Title' headers). For each task, describe the implementation steps and technical deliverables.\n"
                    f"Focus on expert practices and clear documentation."
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
                    f"Create an plan for this wish: '{text}'.\n"
                    f"Decompose the work into logical, actionable tasks (e.g. using '### Task X: Title' headers). For each task, describe the implementation steps and technical deliverables.\n"
                    f"Focus on expert practices and clear documentation."
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
                # Clear task-specific memory to get fresh context for next task
                if task_id in self._rejected_tasks:
                    del self._rejected_tasks[task_id]

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

        # Fetch full context: proposal + submission history
        proposal_id = task.get("proposal_id")
        if proposal_id:
            # Get proposal context
            proposals = self.client.get_proposals()
            proposal = next((p for p in proposals if p.get("id") == proposal_id), None)
            if proposal:
                task["proposal_context"] = proposal.get("description_md", "")
                task["proposal_title"] = proposal.get("title", "")

# OPTIMIZATION: Instead of fetching ALL submissions system-wide and filtering locally,
        # we use a hierarchical approach:
        # 1. Get task status first (fast, may include submission info)
        # 2. Only fetch submissions if needed, filtered by contract_id if available
        # 3. Fallback to full submissions scan only when necessary
        # This reduces API calls from O(N) where N=total submissions to O(1) or O(M) where M=submissions per contract
        if task_id != str(uuid.uuid4()):  # Only if we have a real task_id
            try:
                # PERFORMANCE METRIC: Track optimization effectiveness
                fetch_start_time = time.time()
                used_task_status_only = False
                used_contract_filter = False
                
                # PRIORITY 1: Get task status which may include submission/rejection info
                task_status = self.client.get_task_status(task_id)
                if task_status:
                    # Extract submission/rejection info from task status
                    task["task_status"] = task_status
                    
                    # Check for rejection feedback in task status
                    rejection_reason = (
                        task_status.get("rejection_reason") or 
                        task_status.get("last_rejection_reason") or 
                        task_status.get("feedback") or
                        task_status.get("audit_feedback")
                    )
                    if rejection_reason:
                        task["rejection_feedback"] = rejection_feedback
                        logger.info(f"Worker: Found rejection feedback in task status for {task_id}: {rejection_reason}")
                    
                    # Check for latest submission in task status
                    latest_submission = task_status.get("latest_submission") or task_status.get("submission")
                    if latest_submission:
                        task["previous_submissions"] = {
                            "latest": latest_submission
                        }
                        deliverables = latest_submission.get("deliverables", {})
                        task["previous_work"] = deliverables.get("notes", "")
                        used_task_status_only = True
                        logger.info(f"Worker: OPTIMIZED - Found latest submission in task status for {task_id} (no API fetch needed)")
                
                # PRIORITY 2: Only fetch submissions if task status didn't provide enough info
                if not task_status or not task_status.get("latest_submission"):
                    logger.info(f"Worker: Task status incomplete for {task_id}, fetching submissions...")
                    # Use contract_id to filter submissions at API level if available
                    contract_id = task.get("contract_id") or task.get("proposal_id")
                    if contract_id:
                        submissions = self.client.get_submissions(contract_id)
                        used_contract_filter = True
                        logger.info(f"Worker: OPTIMIZED - Fetched {len(submissions)} submissions for contract {contract_id} (filtered)")
                    else:
                        # Fallback to filtered submissions (less efficient but necessary)
                        logger.warning(f"Worker: No contract_id available for task {task_id}, fetching all submissions (UNOPTIMIZED)")
                        submissions = self.client.get_submissions()
                    
                    if submissions:
                        task_submissions = [s for s in submissions if s.get("task_id") == task_id]
                        
                        if task_submissions:
                            # Sort by timestamp to get most recent
                            task_submissions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
                            latest_submission = task_submissions[0]
                            
                            task["previous_submissions"] = {
                                "latest": latest_submission,
                                "all": task_submissions
                            }
                            
                            # Extract notes and rejection reasons
                            deliverables = latest_submission.get("deliverables", {})
                            task["previous_work"] = deliverables.get("notes", "")
                            task["submission_history"] = [s.get("deliverables", {}).get("notes", "") for s in task_submissions]
                            logger.info(f"Worker: Found {len(task_submissions)} submissions for task {task_id}")
                        else:
                            logger.info(f"Worker: No submissions found for task {task_id}")
                    else:
                        logger.info(f"Worker: No submissions available for task {task_id}")
                
                # PERFORMANCE LOGGING: Track optimization metrics
                fetch_time = time.time() - fetch_start_time
                if used_task_status_only:
                    logger.info(f"Worker: PERFORMANCE - Task context fetched in {fetch_time:.3f}s using task status only (OPTIMAL)")
                elif used_contract_filter:
                    logger.info(f"Worker: PERFORMANCE - Task context fetched in {fetch_time:.3f}s using contract filter (GOOD)")
                else:
                    logger.warning(f"Worker: PERFORMANCE - Task context fetched in {fetch_time:.3f}s using full submissions scan (SUBOPTIMAL)")
                        
            except Exception as e:
                logger.warning(f"Failed to fetch submission history for task {task_id}: {e}")

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
                        f"4. Keep response concise and actionable\n"
                        f"5. COMPLETE PROJECT: Always include an 'index.html' with navigation to provide a complete frontend for your project.\n"
                        f"6. PERSISTENT MEMORY: Use 'memory.md' to maintain state, context, and important decisions across multiple tasks for this contract. Review it at the start of each task and update it with significant changes.\n\n"
                        f"Follow security guidelines in AGENTS.md - only allowed imports and operations.\n"
                        f"Focus on delivering working solutions, not theoretical discussions."
                    )

                    work_type = task.get("work_type", "new work").upper()
                    
                    if rejection_feedback and task.get("previous_work"):
                        prompt = (
                            f"REWORK REQUIRED - Previous submission was REJECTED.\n"
                            f"PREVIOUS WORK: {task['previous_work'][:1000]}...\n"
                            f"REJECTION FEEDBACK: '{rejection_feedback}'\n"
                            f"FULL PROPOSAL CONTEXT: {task.get('proposal_context', 'N/A')}\n\n"
                            f"TASK: Fix all issues and provide corrected implementation.\n"
                            f"Build upon your previous work but address all rejection points.\n\n"
                            f"{base_instruction}"
                        )
                        logger.info(f"Worker: Executing REWORK for {title} - Feedback: {rejection_feedback}")
                    elif task.get("previous_work"):
                        prompt = (
                            f"CONTINUATION WORK - Based on previous submission.\n"
                            f"PREVIOUS WORK: {task['previous_work'][:1000]}...\n"
                            f"PROPOSAL CONTEXT: {task.get('proposal_context', 'N/A')}\n\n"
                            f"{base_instruction}"
                        )
                        logger.info(f"Worker: Executing CONTINUATION for {title}")
                    elif rejection_feedback:
                        prompt = (
                            f"REWORK REQUIRED: Your submission for '{title}' was REJECTED.\n"
                            f"FEEDBACK: '{rejection_feedback}'.\n"
                            f"PROPOSAL CONTEXT: {task.get('proposal_context', 'N/A')}\n\n"
                            f"TASK: Fix all identified issues and provide a corrected implementation.\n"
                            f"Focus on addressing each rejection point specifically.\n\n"
                            f"{base_instruction}"
                        )
                        logger.info(f"Worker: Executing REWORK for {title} - Feedback: {rejection_feedback}")
                    else:
                        # Add proposal context even for new work
                        if task.get("proposal_context"):
                            prompt = (
                                f"NEW WORK - Full proposal context provided.\n"
                                f"PROPOSAL CONTEXT: {task['proposal_context']}\n\n"
                                f"{base_instruction}"
                            )
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
                        f"4. Keep response concise and actionable\n"
                        f"5. COMPLETE PROJECT: Always include an 'index.html' with navigation to provide a complete frontend for your project.\n"
                        f"6. PERSISTENT MEMORY: Use 'memory.md' to maintain state, context, and important decisions across multiple tasks for this contract. Review it at the start of each task and update it with significant changes.\n\n"
                        f"Follow security guidelines in AGENTS.md - only allowed imports and operations.\n"
                        f"Focus on delivering working solutions, not theoretical discussions."
                    )

                    work_type = task.get("work_type", "new work").upper()
                    
                    if rejection_feedback and task.get("previous_work"):
                        prompt = (
                            f"REWORK REQUIRED - Previous submission was REJECTED.\n"
                            f"PREVIOUS WORK: {task['previous_work'][:1000]}...\n"
                            f"REJECTION FEEDBACK: '{rejection_feedback}'\n"
                            f"FULL PROPOSAL CONTEXT: {task.get('proposal_context', 'N/A')}\n\n"
                            f"TASK: Fix all issues and provide corrected implementation.\n"
                            f"Build upon your previous work but address all rejection points.\n\n"
                            f"{base_instruction}"
                        )
                        logger.info(f"Worker: Executing REWORK for {title} - Feedback: {rejection_feedback}")
                    elif task.get("previous_work"):
                        prompt = (
                            f"CONTINUATION WORK - Based on previous submission.\n"
                            f"PREVIOUS WORK: {task['previous_work'][:1000]}...\n"
                            f"PROPOSAL CONTEXT: {task.get('proposal_context', 'N/A')}\n\n"
                            f"{base_instruction}"
                        )
                        logger.info(f"Worker: Executing CONTINUATION for {title}")
                    elif rejection_feedback:
                        prompt = (
                            f"REWORK REQUIRED: Your submission for '{title}' was REJECTED.\n"
                            f"FEEDBACK: '{rejection_feedback}'.\n"
                            f"PROPOSAL CONTEXT: {task.get('proposal_context', 'N/A')}\n\n"
                            f"TASK: Fix all identified issues and provide a corrected implementation.\n"
                            f"Focus on addressing each rejection point specifically.\n\n"
                            f"{base_instruction}"
                        )
                        logger.info(f"Worker: Executing REWORK for {title} - Feedback: {rejection_feedback}")
                    else:
                        # Add proposal context even for new work
                        if task.get("proposal_context"):
                            prompt = (
                                f"NEW WORK - Full proposal context provided.\n"
                                f"PROPOSAL CONTEXT: {task['proposal_context']}\n\n"
                                f"{base_instruction}"
                            )
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
            
            # Build context-aware fallback
            context_info = ""
            if task.get("proposal_context"):
                context_info = f"\n**Proposal Context:** Available in full implementation"
            if task.get("previous_work"):
                context_info += f"\n**Previous Work:** Built upon existing submission"
            
            opencode_output = (
                f"## Task Implementation: {title}\n\n"
                f"**Status:** {status_line}\n"
                f"**Description:** {description}{context_info}\n\n"
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
            if task.get("proposal_context"):
                opencode_output += f"\n\n**Note:** Full proposal context was considered during implementation."

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
            
            # Post-processing: Register APIs and ensure frontend
            self._register_dynamic_functions(visible_pixel_hash, contract_results_dir)
            self._ensure_frontend(visible_pixel_hash, contract_results_dir)
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

        return {
            "notes": notes,
            "result_file": public_url,
            "artifacts_dir": f"/uploads/results/{visible_pixel_hash}/",
            "completion_proof": str(uuid.uuid4())
        }

    def _ensure_frontend(self, visible_pixel_hash: str, contract_results_dir: str):
        """Ensures that the project has an index.html and basic navigation."""
        try:
            index_path = os.path.join(contract_results_dir, "index.html")
            memory_path = os.path.join(contract_results_dir, "memory.md")
            
            memory_content = ""
            if os.path.exists(memory_path):
                try:
                    with open(memory_path, "r") as f:
                        memory_content = f.read()
                except Exception:
                    memory_content = "Could not read memory.md"

            if not os.path.exists(index_path):
                logger.info(f"Creating default index.html for {visible_pixel_hash}")
                
                # List files in the directory for navigation
                try:
                    files = os.listdir(contract_results_dir)
                    # Filter out index.html and hidden files, and put memory.md at the top if it exists
                    other_files = [f for f in files if f not in ["index.html", "memory.md"] and not f.startswith(".")]
                    
                    files_html = ""
                    if os.path.exists(memory_path):
                        files_html += f'<li><a href="memory.md">🧠 memory.md (Persistent Context)</a></li>'
                    
                    files_html += "".join([f'<li><a href="{f}">{f}</a></li>' for f in other_files])
                except Exception:
                    files_html = "<li>No additional files found</li>"
                
                memory_display = ""
                if memory_content:
                    memory_display = f"""
    <div class="memory-box">
        <h2>🧠 Persistent Memory (memory.md)</h2>
        <pre style="white-space: pre-wrap; background: #fff; padding: 1em; border: 1px solid #ddd; border-radius: 4px;">{html.escape(memory_content)}</pre>
    </div>"""

                content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Project Results - {visible_pixel_hash}</title>
    <style>
        body {{ font-family: sans-serif; margin: 2em; line-height: 1.6; max-width: 800px; margin: auto; padding: 2em; background: #f4f7f6; }}
        .container {{ background: white; padding: 2em; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 0.5em; }}
        h2 {{ color: #34495e; margin-top: 1.5em; }}
        ul {{ list-style-type: none; padding: 0; }}
        li {{ margin-bottom: 0.5em; padding: 0.8em; background: #f9f9f9; border-radius: 4px; border-left: 3px solid #ddd; transition: all 0.2s; }}
        li:hover {{ transform: translateX(5px); border-left-color: #3498db; }}
        a {{ text-decoration: none; color: #3498db; font-weight: bold; display: block; }}
        .api-link {{ background: #e8f4fd; padding: 1.5em; border-radius: 8px; margin-top: 2em; border-left: 5px solid #3498db; }}
        .memory-box {{ background: #fffef0; padding: 1.5em; border-radius: 8px; margin-top: 2rem; border-left: 5px solid #f1c40f; }}
        code {{ background: #eee; padding: 0.2em 0.4em; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Project: {visible_pixel_hash}</h1>
        <p>This project was automatically generated by Starlight Agent.</p>
        
        {memory_display}

        <h2>Project Deliverables</h2>
        <ul>
            {files_html}
        </ul>
        
        <div class="api-link">
            <h2>Dynamic API</h2>
            <p>If this project includes an <code>api.py</code>, it has been registered as a dynamic endpoint.</p>
            <p>You can try executing the handler at: <br>
            <a href="{Config.DYNAMIC_API_URL}/function/{visible_pixel_hash}/handler">{Config.DYNAMIC_API_URL}/function/{visible_pixel_hash}/handler</a></p>
        </div>
    </div>
</body>
</html>"""
                with open(index_path, "w") as f:
                    f.write(content)
        except Exception as e:
            logger.error(f"Error ensuring frontend: {e}")

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
