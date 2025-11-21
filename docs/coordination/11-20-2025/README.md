# Dataset Integration Coordination
**Date:** November 20, 2025  
**Goal:** Integrate Grok's negative datasets with Gemini's training pipeline

## Problem Statement
Grok generated 5,000 negative examples for training, but Gemini's training pipeline doesn't use them because:
1. Data is in wrong location (`training/v3_negatives/` vs `negatives/`)
2. Directory structure doesn't match expected schema
3. Training script only scans `clean/` and `stego/` directories

## Solution Overview
Move Grok's negatives to standard location and update Gemini's training to use them.

## Agent Tasks

### Grok (Data Provider)
- [x] Move negatives to `datasets/grok_submission_2025/negatives/`
- [x] Rename directories to match unified schema
- [x] Update manifest to standardized format
- [x] Validate all examples are truly clean

**Details:** `grok_tasks.md`

### Gemini (Training Pipeline)  
- [x] Update training script to scan `datasets/*_submission_*/negatives/`
- [x] Integrate negative examples in training batches
- [x] Ensure 6-stream extraction works with negatives
- [x] Add negative-specific loss weighting

**Details:** `gemini_tasks.md`

### Claude (Schema & Validation)
- [ ] Finalize unified negative schema specification
- [ ] Create validation scripts for dataset integrity
- [ ] Build integration test suite
- [ ] Document process for future agents

**Details:** `claude_tasks.md`

## Timeline

| Day | Grok | Gemini | Claude |
|-----|------|--------|--------|
| Nov 20 | Move/reorganize data | Prepare pipeline changes | Finalize schema |
| Nov 21 | Update manifest/validate | Implement training integration | Create validation tests |
| Nov 22 | Complete validation | Test training with negatives | Final integration validation |

## Success Criteria
- [x] All negative datasets in standard location
- [x] Gemini's training uses negative examples
- [x] Reduced false positives on special cases
- [ ] Clear documentation for future work

## Communication
- **Progress Updates:** Each agent updates daily in `[agent]_progress.md`
- **Blockers:** Report immediately in `blockers.md`
- **Final Sign-off:** All agents in `integration_complete.md`

## Files
- `grok_tasks.md` - Grok's detailed task list
- `gemini_tasks.md` - Gemini's detailed task list  
- `claude_tasks.md` - Claude's detailed task list
- `dataset_integration_plan.md` - Overall coordination plan

---
**Status:** Active - All agents assigned tasks
**Next Review:** November 22, 2025
**Expected Completion:** November 22, 2025
