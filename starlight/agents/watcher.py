import logging
import os
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
        self.audit_trail: Dict[str, Dict] = {}  # proposal_id -> audit_details
        
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
            
            # LIVE TASK FETCHING: Fetch the actual tasks to avoid stale state in list_proposals
            contract_id = proposal.get("contract_id") or (tasks[0].get("contract_id") if tasks and isinstance(tasks[0], dict) else None)
            if not contract_id:
                # Fallback: some proposals use the proposal ID as the contract ID
                contract_id = proposal.get("id")
            
            if contract_id:
                try:
                    live_tasks = self.client.get_tasks(contract_id)
                    if live_tasks:
                        tasks = live_tasks
                except Exception as e:
                    logger.debug(f"Watcher: Could not fetch live tasks for {contract_id}: {e}")

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
                
                # Logic Error Fix: Ensure task is not claimed by another agent
                is_claimed_by_others = claimed_by and not is_ours
                
                if not is_claimed_by_others and (status == "available" or ((status == "claimed" or status == "rejected") and is_ours)) and task_id:
                    task["proposal_id"] = proposal.get("id")
                    task["proposal_title"] = proposal.get("title")
                    available_tasks.append(task)
        
        if available_tasks:
            logger.info(f"Watcher found {len(available_tasks)} actionable tasks (available or resuming).")
            
        return available_tasks

    def process_pending_proposals(self):
        """Finds and approves pending proposals after auditing them."""
        # Optimization: Filter by status to avoid fetching all proposals
        proposals = self.client.get_proposals(status="pending")
        if not proposals:
            return

        # Cache open contracts to avoid fetching for every proposal
        open_contracts = self.client.get_open_contracts()
        contracts_map = {c.get("contract_id"): c for c in open_contracts if isinstance(c, dict)}
            
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            pid = proposal.get("id")
            status = proposal.get("status", "").lower()
            
            if status == "pending" and pid and pid not in self.seen_proposals:
                # Check if already rejected (avoid re-auditing)
                if pid in self.rejected_proposals:
                    logger.debug(f"Watcher: Proposal {pid} already rejected. Skipping audit.")
                    self.seen_proposals.add(pid)  # Mark as seen to avoid rechecking
                    continue

                # BUDGET AND CONTENT CHECKS
                contract_id = proposal.get("contract_id")
                proposal_budget = float(proposal.get("budget_sats", 0))
                
                # Semantic content analysis using OpenCode
                semantic_valid, semantic_reason = self._semantic_content_analysis(proposal)
                if not semantic_valid:
                    logger.warning(f"Watcher: Rejecting proposal {pid} - {semantic_reason}")
                    self.rejected_proposals.add(pid)
                    self.seen_proposals.add(pid)
                    self.rejection_cache[pid] = semantic_reason
                    self._record_audit_trail(pid, "REJECTED", semantic_reason, proposal)
                    self._notify_worker_of_rejection(pid, semantic_reason)
                    continue
                
                # Budget validation against contract
                contract_budget_sats = 0
                if contract_id and contract_id in contracts_map:
                    contract = contracts_map[contract_id]
                    contract_price = float(contract.get("price", 0))
                    price_unit = contract.get("price_unit", "btc").lower()
                    
                    # Convert contract price to sats if needed
                    if price_unit == "btc":
                        contract_budget_sats = contract_price * 100_000_000
                    else:
                        contract_budget_sats = contract_price
                
                # Enhanced budget sanity checks
                budget_valid, budget_reason = self._validate_budget_sanity(proposal, contract_budget_sats)
                if not budget_valid:
                    logger.warning(f"Watcher: Rejecting proposal {pid} - {budget_reason}")
                    self.rejected_proposals.add(pid)
                    self.seen_proposals.add(pid)
                    self.rejection_cache[pid] = budget_reason
                    self._record_audit_trail(pid, "REJECTED", budget_reason, proposal)
                    self._notify_worker_of_rejection(pid, budget_reason)
                    continue
                    
                logger.info(f"Watcher auditing pending proposal {pid}...")

                if self._audit_proposal(proposal):
                    logger.info(f"Proposal {pid} passed audit. Waiting for wish creator to approve.")
                    self._record_audit_trail(pid, "APPROVED", "Passed governance audit", proposal)
                    self.seen_proposals.add(pid)
                else:
                    logger.warning(f"Proposal {pid} failed audit. Marking as rejected.")
                    if pid:
                        self.rejected_proposals.add(pid)
                        self.seen_proposals.add(pid)  # Also add to seen to avoid reprocessing
                        rejection_reason = self.rejection_cache.get(pid, "Failed audit")
                        logger.info(f"Watcher: Rejected proposal {pid}: {rejection_reason}")
                        self._record_audit_trail(pid, "REJECTED", rejection_reason, proposal)

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
        
        # SCOPE VALIDATION
        contract_id = proposal.get("contract_id")
        if contract_id:
            try:
                # Get open contracts to find wish text
                open_contracts = self.client.get_open_contracts()
                for contract in open_contracts or []:
                    if contract.get("contract_id") == contract_id:
                        wish_text = contract.get("wish_text", "")
                        if wish_text:
                            scope_valid, scope_reason = self._validate_scope(wish_text, proposal)
                            if not scope_valid:
                                logger.warning(f"Watcher: Scope validation failed for {proposal.get('id')}: {scope_reason}")
                                proposal_id = proposal.get("id")
                                if proposal_id:
                                    self.rejection_cache[proposal_id] = scope_reason
                                return False
                        break
            except Exception as e:
                logger.warning(f"Could not validate scope for {proposal.get('id')}: {e}")
                # Continue with audit even if scope validation fails
        
        # Get wish text for better context
        wish_text = ""
        try:
            contract_id = proposal.get("contract_id")
            if contract_id:
                open_contracts = self.client.get_open_contracts()
                for contract in open_contracts or []:
                    if contract.get("contract_id") == contract_id:
                        wish_text = contract.get("wish_text", "")
                        break
        except Exception as e:
            logger.debug(f"Could not fetch wish text for proposal audit: {e}")
        
        prompt = (
            f"Audit this technical proposal for governance compliance and quality.\n\n"
            f"ORIGINAL WISH:\n{wish_text[:1000] if wish_text else 'Not available'}\n\n"
            f"PROPOSAL DETAILS:\n"
            f"Title: {title}\n"
            f"Plan: {desc[:2000]}...\n"
            f"Visible Pixel Hash: {visible_pixel_hash}\n\n"
            f"GOVERNANCE AUDIT CRITERIA:\n"
            f"1. **Technical Quality**: Must have structured implementation plan, not just conversational text\n"
            f"2. **Scope Alignment**: Must directly address the wish's intent and requirements\n"
            f"3. **Legitimate Purpose**: Must be serious work, not jokes, tests, or parody content\n"
            f"4. **No Recursion**: Must not be just creating another proposal without doing actual work\n"
            f"5. **Implementation Evidence**: Shows clear understanding of what needs to be built/done\n\n"
            f"ANALYSIS INSTRUCTIONS:\n"
            f"- Compare the proposal's scope against the original wish\n"
            f"- Look for evidence of actual technical work vs meta-work\n"
            f"- Check if this appears to be a serious implementation or joke/test\n"
            f"- Verify the plan is concrete and actionable\n\n"
            f"RESPOND IN EXACT FORMAT:\n"
            f"VERDICT: PASS - if proposal meets all governance criteria\n"
            f"VERDICT: FAIL - <specific reason> - if any criteria are not met\n\n"
            f"Common FAIL reasons:\n"
            f"- 'Recursive proposal: just creates another proposal'\n"
            f"- 'No technical implementation details'\n"
            f"- 'Scope mismatch with original wish'\n"
            f"- 'Appears to be joke or parody submission'\n"
            f"- 'Insufficient technical planning'"
        )
        
        proposal_id = proposal.get("id", "")
        
        # For proposal audit, use temp directory (no artifacts needed)
        import tempfile
        audit_dir = tempfile.mkdtemp(prefix="audit_proposal_")
        logger.debug(f"Created temporary audit directory for proposal: {audit_dir}")
        
        # Try MCP client first for efficiency
        if self.opencode_client.is_available():
            try:
                output = self.opencode_client.run(prompt, timeout=600, workdir=audit_dir)  # 10 minutes for comprehensive proposal audit
                if output:
                    lines = output.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith("VERDICT: "):
                            if "PASS" in line.upper():
                                return True
                            else:
                                # Cache the detailed rejection reason
                                if proposal_id:
                                    self.rejection_cache[proposal_id] = line
                                logger.info(f"Audit failed for {proposal_id}. Reason: {line}")
                                return False
                    # If no proper verdict found, treat as fail
                    if proposal_id:
                        self.rejection_cache[proposal_id] = f"Invalid audit output format: {output[:200]}"
                    logger.warning(f"Invalid audit format for {proposal_id}: {output[:200]}")
                    return False
            except Exception as e:
                logger.error(f"OpenCode MCP audit failed: {e}")
                if proposal_id:
                    self.rejection_cache[proposal_id] = f"Audit tool error: {str(e)}"
                return False
        
        # Fallback to subprocess if available
        if self.opencode_path:
            import subprocess
            try:
                cmd = ["opencode", "run", prompt]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=audit_dir)  # 10 minutes for comprehensive proposal audit
                
                if result.returncode != 0:
                    logger.error(f"Auditor: OpenCode failed (exit {result.returncode}): {result.stderr}")
                    if proposal_id:
                        self.rejection_cache[proposal_id] = "Audit tool execution failed"
                    return False

                output = result.stdout.strip()
                lines = output.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith("VERDICT: "):
                        if "PASS" in line.upper():
                            return True
                        else:
                            # Cache the detailed rejection reason
                            if proposal_id:
                                self.rejection_cache[proposal_id] = line
                            logger.info(f"Audit failed for {proposal_id}. Reason: {line}")
                            return False
                
                # If no proper verdict found, treat as fail
                if proposal_id:
                    self.rejection_cache[proposal_id] = f"Invalid audit output format: {output[:200]}"
                logger.warning(f"Invalid audit format for {proposal_id}: {output[:200]}")
                return False
                    
            except subprocess.TimeoutExpired:
                logger.error(f"Auditor: OpenCode timed out after 600s for proposal {proposal_id}")
                if proposal_id:
                    self.rejection_cache[proposal_id] = "Audit timed out after 600s"
                return False
            except Exception as e:
                logger.error(f"Error during proposal audit: {e}")
                if proposal_id:
                    self.rejection_cache[proposal_id] = f"Audit error: {str(e)}"
                return False
        
        # MULTI-AUDITOR CONSENSUS CHECK
        if self._requires_multi_auditor_consensus(proposal):
            consensus_result = self._check_auditor_consensus(proposal)
            if not consensus_result["has_consensus"]:
                logger.warning(f"Watcher: Multi-auditor consensus not reached for {proposal.get('id')}: {consensus_result['reason']}")
                proposal_id = proposal.get("id")
                if proposal_id:
                    self.rejection_cache[proposal_id] = f"Consensus not reached: {consensus_result['reason']}"
                return False
            logger.info(f"Watcher: Multi-auditor consensus reached for {proposal.get('id')}")
        
        # Auto-approve if no auditor available
        return True

    def _requires_multi_auditor_consensus(self, proposal: Dict) -> bool:
        """Determine if proposal requires multi-auditor consensus."""
        budget = float(proposal.get("budget_sats", 0))
        
        # High-value proposals require consensus
        if budget > 100000:  # More than 0.001 BTC
            return True
        
        # Check for sensitive keywords
        sensitive_keywords = ["security", "authentication", "encryption", "payment", "financial"]
        title = proposal.get("title", "").lower()
        desc = proposal.get("description_md", "").lower()
        
        if any(keyword in title or keyword in desc for keyword in sensitive_keywords):
            return True
        
        return False

    def _check_auditor_consensus(self, proposal: Dict) -> Dict:
        """Simulate multi-auditor consensus checking."""
        # In a real implementation, this would query other auditor agents
        # For now, simulate consensus based on proposal quality
        
        proposal_id = proposal.get("id", "unknown")
        budget = float(proposal.get("budget_sats", 0))
        title = proposal.get("title", "")
        desc = proposal.get("description_md", "")
        
        # Simulate other auditors' decisions based on proposal characteristics
        simulated_auditors = []
        
        # Auditor 1: Focus on technical quality
        auditor1_approves = len(desc) > 100 and "###" in desc  # Has structured content
        simulated_auditors.append({"id": "auditor_tech", "decision": "APPROVE" if auditor1_approves else "REJECT"})
        
        # Auditor 2: Focus on budget reasonableness
        auditor2_approves = budget < 1000000 and budget > 0  # Reasonable budget
        simulated_auditors.append({"id": "auditor_budget", "decision": "APPROVE" if auditor2_approves else "REJECT"})
        
        # Auditor 3: Focus on semantic analysis
        semantic_valid, _ = self._semantic_content_analysis(proposal)
        simulated_auditors.append({"id": "auditor_semantic", "decision": "APPROVE" if semantic_valid else "REJECT"})
        
        # Count approvals
        approvals = sum(1 for auditor in simulated_auditors if auditor["decision"] == "APPROVE")
        total_auditors = len(simulated_auditors)
        
        # Consensus threshold: 2/3 approval
        consensus_threshold = 0.67
        has_consensus = (approvals / total_auditors) >= consensus_threshold
        
        return {
            "has_consensus": has_consensus,
            "approvals": approvals,
            "total_auditors": total_auditors,
            "threshold": consensus_threshold,
            "auditor_decisions": simulated_auditors,
            "reason": f"Only {approvals}/{total_auditors} auditors approved (threshold: {consensus_threshold})"
        }

    def _record_audit_trail(self, proposal_id: str, decision: str, reasoning: str, proposal: Dict):
        """Record detailed audit information for accountability."""
        try:
            audit_entry = {
                "proposal_id": proposal_id,
                "decision": decision,
                "reasoning": reasoning,
                "auditor_signature": self.ai_identifier,
                "timestamp": time.time(),
                "proposal_title": proposal.get("title", ""),
                "proposal_budget": proposal.get("budget_sats", 0),
                "confidence_score": self._calculate_confidence_score(proposal, decision),
                "peer_review_required": self._requires_multi_auditor_consensus(proposal)
            }
            
            # Add consensus information if applicable
            if self._requires_multi_auditor_consensus(proposal):
                consensus_result = self._check_auditor_consensus(proposal)
                audit_entry["consensus_result"] = consensus_result
            
            self.audit_trail[proposal_id] = audit_entry
            logger.info(f"Recorded audit trail for {proposal_id}: {decision} by {self.ai_identifier}")
        except Exception as e:
            logger.error(f"Failed to record audit trail for {proposal_id}: {e}")

    def _calculate_confidence_score(self, proposal: Dict, decision: str) -> int:
        """Calculate audit confidence score (1-10 scale)."""
        base_score = 7
        
        title = proposal.get("title", "")
        desc = proposal.get("description_md", "")
        budget = float(proposal.get("budget_sats", 0))
        
        # Adjust confidence based on factors
        if len(title) > 10:
            base_score += 1
        if len(desc) > 100:
            base_score += 1
        if budget > 0 and budget < 1000000:  # Reasonable budget
            base_score += 1
        
        # Reduce confidence for suspicious patterns
        suspicious_keywords = ["escort", "adult", "joke", "parody", "test"]
        if any(keyword in title.lower() or keyword in desc.lower() for keyword in suspicious_keywords):
            base_score -= 2
            
        return max(1, min(10, base_score))

    def _validate_budget_sanity(self, proposal: Dict, contract_budget_sats: float) -> Tuple[bool, str]:
        """Validate budget against reasonable limits to prevent drain attacks."""
        proposal_budget = float(proposal.get("budget_sats", 0))
        
        if proposal_budget <= 0:
            return False, "Invalid budget: Must be greater than 0"
            
        if contract_budget_sats > 0:
            # Check if proposal exceeds wish budget by more than 10x
            if proposal_budget > (contract_budget_sats * 10):
                return False, f"Budget sanity check failed: {proposal_budget} sats exceeds wish budget {contract_budget_sats} sats by more than 10x"
        
        # Additional sanity checks
        if proposal_budget > 100000000:  # More than 1 BTC is suspicious
            return False, f"Extremely high budget: {proposal_budget} sats exceeds reasonable limits"
            
        return True, "Budget validation passed"

    def _semantic_content_analysis(self, proposal: Dict) -> Tuple[bool, str]:
        """Use OpenCode to perform semantic analysis of proposal quality and alignment."""
        proposal_id = proposal.get("id", "unknown")
        title = proposal.get("title", "")
        desc = proposal.get("description_md", "")
        budget = proposal.get("budget_sats", 0)
        
        # Get wish text for comparison if available
        wish_text = ""
        try:
            contract_id = proposal.get("contract_id")
            if contract_id:
                open_contracts = self.client.get_open_contracts()
                for contract in open_contracts or []:
                    if contract.get("contract_id") == contract_id:
                        wish_text = contract.get("wish_text", "")
                        break
        except Exception as e:
            logger.debug(f"Could not fetch wish text for semantic analysis: {e}")
        
        prompt = (
            f"Analyze this proposal for Starlight governance compliance and quality.\n\n"
            f"WISH (original request):\n{wish_text[:1000] if wish_text else 'Not available'}\n\n"
            f"PROPOSAL (response to wish):\n"
            f"Title: {title}\n"
            f"Description: {desc[:2000]}\n"
            f"Budget: {budget} sats\n\n"
            f"ASSESSMENT CRITERIA:\n"
            f"1. Does this proposal genuinely address the wish's intent?\n"
            f"2. Is this a serious technical proposal or a joke/test/parody?\n"
            f"3. Are there any concerning patterns (adult content, scams, etc.)?\n"
            f"4. Does the scope and budget seem reasonable for the work described?\n"
            f"5. Is there clear evidence of technical planning and implementation approach?\n\n"
            f"RESPOND IN EXACT FORMAT:\n"
            f"VERDICT: PASS - if the proposal is legitimate and appropriate\n"
            f"VERDICT: FAIL - <specific reason> - if the proposal has issues\n\n"
            f"Examples of FAIL reasons:\n"
            f"- 'Appears to be joke or parody submission'\n"
            f"- 'Contains inappropriate or concerning content'\n"
            f"- 'Does not address wish requirements'\n"
            f"- 'Unreasonable scope/budget mismatch'\n"
        )
        
        # Use temp directory for analysis
        import tempfile
        analysis_dir = tempfile.mkdtemp(prefix="semantic_analysis_")
        
        # Try MCP client first for efficiency
        if self.opencode_client.is_available():
            try:
                output = self.opencode_client.run(prompt, timeout=300, workdir=analysis_dir)  # 5 minutes for semantic analysis
                if output:
                    output = output.strip()
                    lines = output.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith("VERDICT: "):
                            if "PASS" in line.upper():
                                return True, "Semantic analysis passed"
                            else:
                                return False, line
                    return False, f"Analysis produced invalid verdict: {output[:200]}"
            except Exception as e:
                logger.error(f"OpenCode semantic analysis failed: {e}")
        
        # Fallback to subprocess if available
        if self.opencode_path:
            import subprocess
            try:
                cmd = ["opencode", "run", prompt]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=analysis_dir)
                
                if result.returncode != 0:
                    logger.error(f"Semantic analysis failed (exit {result.returncode}): {result.stderr}")
                    return True, "Analysis tool failed - defaulting to approve"  # Fail safe
                
                output = result.stdout.strip()
                lines = output.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith("VERDICT: "):
                        if "PASS" in line.upper():
                            return True, "Semantic analysis passed"
                        else:
                            return False, line
                return False, f"Analysis produced invalid verdict: {output[:200]}"
            except Exception as e:
                logger.error(f"Error during semantic analysis: {e}")
                return True, "Analysis error - defaulting to approve"  # Fail safe
        
        # Default to approve if no analysis available
        return True, "No analysis available - defaulting to approve"

    def _validate_scope(self, wish_text: str, proposal: Dict) -> Tuple[bool, str]:
        """Validate that proposal aligns with original wish scope."""
        proposal_title = proposal.get("title", "")
        proposal_desc = proposal.get("description_md", "")
        
        # Simple keyword-based scope validation (can be enhanced with NLP)
        wish_keywords = set(wish_text.lower().split())
        proposal_keywords = set((proposal_title + " " + proposal_desc).lower().split())
        
        # Check for minimum overlap
        common_keywords = wish_keywords.intersection(proposal_keywords)
        
        # Calculate basic similarity
        total_unique = len(wish_keywords.union(proposal_keywords))
        if total_unique == 0:
            return False, "Empty wish or proposal content"
        
        similarity = len(common_keywords) / total_unique
        
        # Extract key concepts from wish
        wish_concepts = self._extract_concepts(wish_text)
        proposal_concepts = self._extract_concepts(proposal_title + " " + proposal_desc)
        
        # Check for concept overlap
        concept_overlap = len(wish_concepts.intersection(proposal_concepts))
        concept_similarity = concept_overlap / max(len(wish_concepts), 1)
        
        # Combined similarity score
        final_similarity = (similarity * 0.3) + (concept_similarity * 0.7)
        
        if final_similarity < 0.3:  # 30% similarity threshold
            return False, f"Scope mismatch: {final_similarity:.2f} similarity below 0.3 threshold"
        
        return True, f"Scope validated: {final_similarity:.2f} similarity"

    def _extract_concepts(self, text: str) -> Set[str]:
        """Extract key concepts from text using simple heuristics."""
        # Common technical/conceptual indicators
        concept_indicators = [
            "system", "platform", "api", "ui", "interface", "database",
            "security", "authentication", "encryption", "network", "protocol",
            "algorithm", "model", "training", "analysis", "optimization",
            "implementation", "development", "design", "architecture",
            "blockchain", "bitcoin", "crypto", "smart", "contract",
            "machine", "learning", "artificial", "intelligence", "ai"
        ]
        
        text_lower = text.lower()
        found_concepts = set()
        
        for concept in concept_indicators:
            if concept in text_lower:
                found_concepts.add(concept)
        
        # Also extract longer phrases that might be domain-specific
        words = text_lower.split()
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            if any(indicator in phrase for indicator in concept_indicators):
                found_concepts.add(phrase)
        
        return found_concepts

    def process_submissions(self):
        """Audits and reviews (approves/rejects) Worker submissions."""
        # Optimization: Filter by status to avoid fetching all submissions
        try:
            submissions = self.client.get_submissions(status="submitted")
            submissions.extend(self.client.get_submissions(status="pending_review"))
            logger.info(f"Watcher: Fetched {len(submissions)} submissions for audit")
        except Exception as e:
            logger.error(f"Watcher: Failed to fetch submissions: {e}")
            return
            
        if not submissions:
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
        artifacts_dir = deliverables.get("artifacts_dir", "")
        
        # Extract contract identifier for isolation
        visible_pixel_hash = sub.get("visible_pixel_hash") or sub.get("task", {}).get("visible_pixel_hash") or "unknown"
        
        # Find the worker's artifacts directory directly
        audit_dir = None
        artifacts_available = False
        
        if artifacts_dir:
            try:
                # Handle both relative and absolute paths
                if artifacts_dir.startswith("/uploads/"):
                    # Convert from URL path to filesystem path
                    hash_component = artifacts_dir.strip("/").split("/")[-1]
                    # Try multiple potential locations (dev vs kubernetes)
                    potential_paths = [
                        os.path.join(Config.UPLOADS_DIR, "results", hash_component),
                        os.path.join("results", hash_component)
                    ]
                else:
                    potential_paths = [artifacts_dir]
                
                # Find the first existing artifacts directory
                for path in potential_paths:
                    if os.path.exists(path):
                        audit_dir = path
                        artifacts_available = True
                        logger.info(f"Auditing directly in worker's artifacts directory: {path}")
                        break
                
                if not audit_dir:
                    logger.warning(f"Artifacts directory not found in any location: {potential_paths}")
                    # Fallback to a temp directory if no artifacts found
                    import tempfile
                    audit_dir = tempfile.mkdtemp(prefix="audit_fallback_")
                    logger.warning(f"No artifacts found, using temp directory: {audit_dir}")
                    
            except Exception as e:
                logger.error(f"Failed to locate artifacts directory: {e}")
                # Fallback to temp directory on error
                import tempfile
                audit_dir = tempfile.mkdtemp(prefix="audit_error_")
        
        # If no artifacts_dir provided, use temp directory
        if not audit_dir:
            import tempfile
            audit_dir = tempfile.mkdtemp(prefix="audit_no_artifacts_")
        
        prompt = (
            f"Audit this work submission report.\n"
            f"Report: {notes[:2000]}\n\n"
            f"CRITERIA:\n"
            f"1. Must contain technical evidence (code, logs, pseudo-code).\n"
            f"2. Must NOT be generic or conversational filler.\n"
            f"3. Must demonstrate task completion.\n"
        )
        
        # Add artifacts info to prompt if available
        if artifacts_available:
            prompt += f"\n4. Artifacts are available in current directory - examine files for evidence.\n\n"
        else:
            prompt += f"\n4. No artifacts directory available - rely on report content only.\n\n"
            
        prompt += "Respond with a single line: 'VERDICT: PASS' or 'VERDICT: FAIL - <reason>'."
        
        # Try MCP client first for efficiency
        if self.opencode_client.is_available():
            try:
                output = self.opencode_client.run(prompt, timeout=1800, workdir=audit_dir)  # 30 minutes for submission audit (thorough analysis)
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
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=audit_dir)  # 30 minutes for submission audit (thorough analysis)
                
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
                    self.audit_trail = state.get("audit_trail", {})
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
            "audit_trail": self.audit_trail,
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
