import logging
import time
import subprocess
import shutil
from typing import List, Dict, Set, Optional, Any
from .client import StargateClient
from .config import Config

logger = logging.getLogger(__name__)

class WatcherAgent:
    def __init__(self, client: StargateClient, ai_identifier: str = Config.AI_IDENTIFIER):
        self.client = client
        self.ai_identifier = ai_identifier
        self.seen_tasks: Set[str] = set()
        self.seen_proposals: Set[str] = set()
        self.seen_submissions: Set[str] = set()
        self.opencode_path = shutil.which("opencode")
        if self.opencode_path:
            logger.info(f"Watcher: OpenCode detected at {self.opencode_path}. Auditing enabled.")
        else:
            logger.warning("Watcher: OpenCode not found. Falling back to auto-approval.")

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
                
                if (status == "available" or (status == "claimed" and is_ours)) and task_id not in self.seen_tasks:
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
                logger.info(f"Watcher auditing pending proposal {pid}...")
                
                if self._audit_proposal(proposal):
                    logger.info(f"Proposal {pid} passed audit. Approving...")
                    if self.client.approve_proposal(pid):
                        logger.info(f"Proposal {pid} approved.")
                        self.seen_proposals.add(pid)
                else:
                    logger.warning(f"Proposal {pid} failed audit. Skipping approval.")

    def _audit_proposal(self, proposal: Dict) -> bool:
        """Uses OpenCode to decide if a proposal should be approved."""
        if not self.opencode_path:
            return True # Auto-approve if no auditor available
            
        title = proposal.get("title", "")
        desc = proposal.get("description_md", "")
        prompt = (
            f"Audit this technical proposal.\n"
            f"Title: {title}\n"
            f"Plan: {desc[:2000]}...\n\n"
            f"CRITERIA:\n"
            f"1. Must have structured tasks using '### Task X: Title' headers.\n"
            f"2. Must have a technical, non-conversational plan.\n"
            f"3. Must be relevant to the title.\n\n"
            f"INSTRUCTION:\n"
            f"Analyze the plan and decide if it meets the criteria. "
            f"Respond with a single line: 'VERDICT: PASS' or 'VERDICT: FAIL - <reason>'."
        )
        
        try:
            cmd = ["opencode", "run", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"Auditor: OpenCode failed (exit {result.returncode}): {result.stderr}")
                return False

            output = result.stdout.strip().upper()
            
            # Robust verdict parsing: check if OpenCode explicitly said PASS
            # We trust the AI to have validated the structure.
            if "VERDICT: PASS" in output or "PASS" in output.split()[:10]:
                return True
                
            logger.info(f"Audit failed for {proposal.get('id')}. Auditor output: {output}")
            return False
        except subprocess.TimeoutExpired:
            logger.error(f"Auditor: OpenCode timed out after 300s for proposal {proposal.get('id')}")
            return False
        except Exception as e:
            logger.error(f"Error during proposal audit: {e}")
            return False # Safety: don't approve if tool crashes

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
                    if self.client.review_submission(sub_id, "approve"):
                        logger.info(f"Submission {sub_id} approved.")
                        self.seen_submissions.add(sub_id)
                else:
                    # audit_passed is False, 'reason' contains the Auditor's critique
                    logger.warning(f"Submission {sub_id} failed audit: {reason}")
                    if self.client.review_submission(sub_id, "reject", reason):
                        logger.info(f"Submission {sub_id} rejected with feedback.")
                        self.seen_submissions.add(sub_id)

    def _audit_submission(self, sub: Dict) -> (bool, str):
        """Uses OpenCode to verify the quality of a submission."""
        if not self.opencode_path:
            return True, ""
            
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
        
        try:
            cmd = ["opencode", "run", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
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

    def run_once(self) -> List[Dict]:
        try:
            self.process_pending_proposals()
            self.process_submissions()
            return self.find_available_tasks()
        except Exception as e:
            logger.error(f"Watcher encountered error: {e}")
            return []
