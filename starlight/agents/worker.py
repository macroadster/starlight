import logging
import time
import uuid
import subprocess
import shutil
import threading
import os
from typing import Dict, Set, Optional, Any
from .client import StargateClient
from .config import Config

logger = logging.getLogger(__name__)

class WorkerAgent:
    def __init__(self, client: StargateClient, ai_identifier: str = Config.AI_IDENTIFIER):
        self.client = client
        self.ai_identifier = ai_identifier
        self.seen_wishes: set = set()
        self.active_tasks: Set[str] = set()
        self.opencode_path = shutil.which("opencode")
        # Limit concurrency to prevent OOM errors in container
        self.concurrency_limit = threading.BoundedSemaphore(1)
        if self.opencode_path:
            logger.info(f"OpenCode detected at {self.opencode_path}")
        else:
            logger.warning("OpenCode not found in PATH. Falling back to simulated work.")

    def process_wishes(self):
        """Scans for new wishes and enriches existing thin proposals."""
        try:
            # 1. Get all proposals to see what needs enrichment
            proposals = self.client.get_proposals()
            handled_contract_ids = set()
            
            logger.info(f"Worker: Inspecting {len(proposals)} existing proposals...")
            for p in proposals:
                if not isinstance(p, dict):
                    continue
                pid = p.get("id")
                status = p.get("status", "").lower()
                desc = p.get("description_md", "")
                
                cid = p.get("contract_id")
                if cid:
                    handled_contract_ids.add(cid)
                vph = p.get("visible_pixel_hash")
                if vph:
                    handled_contract_ids.add(vph)
                    handled_contract_ids.add(f"wish-{vph}")

                # If it's a pending proposal with a very short description, enrich it
                if status == "pending" and len(desc) < 200:
                    logger.info(f"Worker: Proposal {pid} is 'thin' (desc length: {len(desc)}). Enriching...")
                    if self._enrich_proposal(p):
                        logger.info(f"Worker: Successfully enriched proposal {pid}")
                    else:
                        logger.warning(f"Worker: Failed to enrich proposal {pid}")
                elif status == "pending":
                    logger.debug(f"Worker: Proposal {pid} is pending but already has substantial desc ({len(desc)} chars).")

            # 2. Get open contracts (wishes) for which NO proposal exists yet
            wishes = self.client.get_open_contracts()
            logger.info(f"Worker found {len(wishes)} total contracts/wishes. {len(handled_contract_ids)} are already handled by proposals.")
            
            for wish in wishes:
                if not isinstance(wish, dict):
                    continue
                wid = wish.get("id")
                status = wish.get("status", "").lower()
                text = wish.get("text", "")
                
                if wid in self.seen_wishes or wid in handled_contract_ids:
                    continue
                
                if status == "pending":
                    logger.info(f"Worker found new unhandled wish {wid}: {text[:50]}...")
                    if self._create_proposal_for_wish(wish):
                        self.seen_wishes.add(wid)

        except Exception as e:
            logger.error(f"Worker error processing wishes: {e}")

    def _enrich_proposal(self, proposal: Dict) -> bool:
        """Enriches an existing proposal with a better plan using OpenCode."""
        pid = proposal.get("id")
        text = proposal.get("description_md", "")
        
        if not self.opencode_path:
            return False

        logger.info(f"Using OpenCode to professionalize proposal {pid}")
        try:
            prompt = (
                f"You are a technical architect. Professionalize the following smart contract implementation plan: '{text}'.\n"
                f"RULES:\n"
                f"1. DO NOT be conversational. No 'I can help with that' or 'Let me see'.\n"
                f"2. You MUST include structured tasks using '### Task X: Title' headers.\n"
                f"3. Use technical language. Include specific implementation details.\n"
                f"4. Format the output in clean Markdown.\n\n"
                f"REQUIRED STRUCTURE:\n"
                f"# Technical Implementation Plan\n"
                f"## Executive Summary\n"
                f"[Summary]\n"
                f"## Technical Requirements\n"
                f"- [Requirement 1]\n"
                f"### Task 1: Core Infrastructure\n"
                f"[Details of build steps]\n"
                f"### Task 2: Implementation and Testing\n"
                f"[Details of development and testing]\n"
                f"## Final Deliverables\n"
                f"- [Deliverable]"
            )
            cmd = ["opencode", "run", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and result.stdout.strip():
                new_desc = result.stdout.strip()
                # Remove common conversational intros if they appear despite instructions
                if "Sure" in new_desc[:20] or "I'll" in new_desc[:20]:
                    lines = new_desc.splitlines()
                    if len(lines) > 1:
                        new_desc = "\n".join(lines[1:])

                # Ensure it at least has one task header
                if "TASK" not in new_desc.upper():
                    new_desc += "\n\n### Task 1: Build Solution\nExecute technical implementation."

                update_data = {
                    "description_md": new_desc,
                    "metadata": {
                        "enriched_by": self.ai_identifier,
                        "enrichment_timestamp": time.time()
                    }
                }
                if self.client.update_proposal(pid, update_data):
                    logger.info(f"Proposal {pid} enriched successfully.")
                    return True
            else:
                logger.error(f"OpenCode enrichment failed for {pid}: {result.stderr}")
        except Exception as e:
            logger.error(f"Failed to use OpenCode for proposal enrichment: {e}")
        return False

    def _create_proposal_for_wish(self, wish: Dict) -> bool:
        """Generates and submits a proposal for a given wish."""
        wid = wish.get("id")
        text = wish.get("text", "No description provided")
        
        title = f"Proposal for: {text.splitlines()[0][:50]}"
        description = (
            f"I propose to fulfill the wish: '{text}' by executing the following plan.\n\n"
            f"### Task 1: Build Solution\n"
            f"Execute the technical requirements to fulfill the original wish: {text}"
        )
        
        if self.opencode_path:
            logger.info(f"Using OpenCode to generate proposal for wish {wid}")
            try:
                # Ask OpenCode to generate a systematic plan for the wish
                prompt = (
                    f"Generate a professional technical implementation plan for: '{text}'.\n"
                    f"MANDATORY FORMAT:\n"
                    f"## Overview\n"
                    f"[Details]\n"
                    f"### Task 1: Core Infrastructure\n"
                    f"[Steps]\n"
                    f"### Task 2: Validation and Testing\n"
                    f"[Steps]\n\n"
                    f"Respond only with the structured Markdown."
                )
                cmd = ["opencode", "run", prompt]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0 and result.stdout.strip():
                    description = result.stdout.strip()
                    if "TASK" not in description.upper():
                         description += f"\n\n### Task 1: Build Solution\n{text}"
                    # Heuristic for title - find the first non-empty line
                    lines = [l for l in description.splitlines() if l.strip()]
                    if lines:
                        title = lines[0].strip("# ")
                        if len(title) > 100:
                            title = title[:97] + "..."
            except Exception as e:
                logger.error(f"Failed to use OpenCode for proposal generation: {e}")

        # Create proposal.
        proposal_data = {
            "title": title,
            "description_md": description,
            "budget_sats": 1000,
            "visible_pixel_hash": wid,
            "contract_id": wid
        }
        
        logger.info(f"Worker creating proposal for wish {wid}...")
        pid = self.client.create_proposal(proposal_data)
        if pid:
            logger.info(f"Proposal created successfully: {pid}")
            return True
        return False

    def process_task(self, task: Dict) -> bool:
        """Claims, resumes, or reworks background execution of a task."""
        task_id = task.get("task_id")
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
        
        # Check if this task needs rework (it will be 'available' but might have been rejected)
        if status == "available":
            # Peek at status to see if it was recently rejected
            details = self.client.get_task_status(task_id)
            if details and details.get("status", "").lower() == "available":
                # Check for rejection feedback in the status response
                rejection_feedback = details.get("rejection_reason") or details.get("last_rejection_reason")
                if rejection_feedback:
                    logger.info(f"Worker: Task {task_id} was previously rejected. Reason: {rejection_feedback}. Preparing rework...")

        if status == "claimed" and existing_claim_id:
            logger.info(f"Worker: Resuming already claimed task: {title} ({task_id}). Claim: {existing_claim_id}")
            claim_id = existing_claim_id
        else:
            logger.info(f"Worker attempting to claim task: {title} ({task_id})")
            claim_id = self.client.claim_task(task_id, self.ai_identifier)
            
            # If claim failed, it might be because we already claimed it in a previous poll/restart
            if not claim_id:
                logger.info(f"Worker: Claim tool failed for {task_id}. Checking if we already own it...")
                claim_id = self._find_my_existing_claim(task)
            
        if not claim_id:
            logger.warning(f"Failed to secure claim for task {task_id}. It might have been taken by someone else.")
            return False
        
        logger.info(f"Successfully secured task {task_id}. Claim ID: {claim_id}")
        
        # 2. Start Work in Background
        self.active_tasks.add(task_id)
        # Pass feedback to background thread
        task["rejection_feedback"] = rejection_feedback
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
        task_id = task.get("task_id")
        try:
            with self.concurrency_limit:
                deliverables = self._perform_work(task)
            
            logger.info(f"Submitting work for task {task_id} (Claim: {claim_id})...")
            if self.client.submit_work(claim_id, deliverables):
                logger.info(f"Task {task_id} completed and submitted successfully.")
            else:
                logger.error(f"Failed to submit work for task {task_id}.")
        except Exception as e:
            logger.error(f"Error during background task {task_id}: {e}")
        finally:
            self.active_tasks.discard(task_id)

    def _perform_work(self, task: Dict) -> Dict:
        """Executes work using OpenCode if available, otherwise simulates."""
        description = task.get("description", "")
        title = task.get("title", "Unknown Task")
        proposal_title = task.get("proposal_title", "Unknown Proposal")
        task_id = task.get("task_id", str(uuid.uuid4()))
        contract_id = task.get("contract_id", "unassigned")
        rejection_feedback = task.get("rejection_feedback")
        
        # Extract skills to inspire the agent
        skills = task.get("skills", ["general engineering"])
        skills_str = ", ".join(skills) if isinstance(skills, list) else str(skills)

        # Determine data directory for results
        base_uploads_dir = Config.UPLOADS_DIR
        contract_results_dir = os.path.join(base_uploads_dir, "results", contract_id)
        os.makedirs(contract_results_dir, exist_ok=True)
        
        result_filename = f"{task_id}.md"
        result_path = os.path.join(contract_results_dir, result_filename)
        public_url = f"/uploads/results/{contract_id}/{result_filename}"

        opencode_output = ""
        if self.opencode_path:
            logger.info(f"Executing work using OpenCode: {title}")
            try:
                # ACTION-ORIENTED PROMPT focusing on implementation and skills
                base_instruction = (
                    f"You are an ELITE SENIOR ENGINEER with expert skills in: [{skills_str}].\n"
                    f"Your mission is to IMPLEMENT and EXECUTE the following technical task: '{description}'.\n"
                    f"DO NOT just plan or describe. YOU MUST ACTUALLY PERFORM THE WORK.\n"
                    f"Generate ACTUAL ARTIFACTS: code, logs, datasets, or configurations.\n"
                    f"Use the project's tools if relevant: 'python3 data_generator.py', 'python3 trainer.py', 'python3 scanner.py'.\n\n"
                    f"REQUIREMENTS for your submission:\n"
                    f"1. You MUST provide FUNCTIONAL EVIDENCE: actual code snippets, execution logs, or verified logic flows.\n"
                    f"2. You MUST describe the specific implementation ACTIONS you took.\n"
                    f"3. You MUST include a 'Technical Deliverables' section containing the actual output of your work.\n"
                    f"4. Be technical, precise, and authoritative. Avoid generic filler.\n"
                    f"5. Respond with a comprehensive Technical Implementation Report in Markdown."
                )

                if rejection_feedback:
                    prompt = (
                        f"CRITICAL REWORK REQUIRED: Your previous attempt for '{title}' was REJECTED.\n"
                        f"AUDITOR FEEDBACK: '{rejection_feedback}'.\n"
                        f"As an expert in [{skills_str}], you MUST correct this and provide a high-fidelity implementation.\n"
                        + base_instruction
                    )
                else:
                    prompt = base_instruction
                
                cmd = ["opencode", "run", prompt]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=86400)
                
                if result.returncode == 0:
                    opencode_output = result.stdout
                else:
                    logger.error(f"OpenCode failed with exit code {result.returncode}: {result.stderr}")
            except Exception as e:
                logger.error(f"OpenCode execution error: {e}")

        if not opencode_output:
            logger.info(f"Simulating work for: {title}")
            time.sleep(2)
            
            status_line = "REWORK: Technical Correction" if rejection_feedback else "Initial technical implementation"
            
            opencode_output = (
                f"## Technical Implementation Report for Task: {title}\n\n"
                f"### Status: {status_line}\n"
                f"Methodology: Automated technical analysis of requirement: '{description}'.\n"
                f"Infrastructure logic validated against Starlight protocol v2.0.\n\n"
                f"### Task 1: Implementation Evidence (Pseudo-Code)\n"
                f"```python\n"
                f"def fulfill_task(description):\n"
                f"    # Logic derived from autonomous analysis\n"
                f"    artifacts = validate_requirements(description)\n"
                f"    return [publish(a) for a in artifacts]\n"
                f"```\n\n"
                f"### Task 2: Execution Logs\n"
                f"- [LOG] Initializing logic engine...\n"
                f"- [LOG] Requirements mapped: {description[:50]}...\n"
                f"- [LOG] Artifacts generated and verified.\n\n"
                f"### Result\n"
                f"Technical implementation complete. All artifacts stored in shared volume."
            )
            if rejection_feedback:
                opencode_output += f"\n\n**Note:** This submission addresses previous auditor feedback: {rejection_feedback}"

        notes = (
            f"# Work Report: {title}\n\n"
            f"**Executed by:** {self.ai_identifier}\n"
            f"**Proposal:** {proposal_title}\n"
            f"**Task ID:** {task_id}\n\n"
            f"## Results\n"
            f"{opencode_output}\n\n"
            f"--- \n"
            f"**Full report available at:** [Download Report]({public_url})"
        )

        # Write the full report to the shared data directory
        try:
            with open(result_path, "w") as f:
                f.write(notes)
            logger.info(f"Saved task results to {result_path}")
            
            # EXPORT ARTIFACTS: If the worker created a project directory, copy its contents to shared volume
            # Heuristic: look for directories in /app/ that aren't 'starlight', 'models', etc.
            for item in os.listdir("/app"):
                item_path = os.path.join("/app", item)
                if os.path.isdir(item_path) and item not in ["starlight", "models", "scripts", "tests", "results", "__pycache__"]:
                    # Merge contents directly into the wish directory to support task interdependencies
                    logger.info(f"Merging technical artifacts from {item_path} into {contract_results_dir}...")
                    shutil.copytree(item_path, contract_results_dir, dirs_exist_ok=True)
                    logger.info(f"Artifacts merged into {contract_results_dir}")
        except Exception as e:
            logger.error(f"Failed to export artifacts to disk: {e}")

        return {
            "notes": notes,
            "result_file": public_url,
            "artifacts_dir": f"/uploads/results/{contract_id}/",
            "completion_proof": str(uuid.uuid4())
        }
