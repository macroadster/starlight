# AI Agent Context for Project Starlight

This document provides the essential context for AI agents working on Project Starlight.

## 1. Core Mission & Objective

**Project Starlight** is an open-source protocol to build and train AI models for detecting steganography in images stored on blockchains like Bitcoin.

- **Primary Goal**: Safeguard the integrity of digital history stored on-chain.
- **Long-Term Vision (2142)**: Automate covert data detection to enable AI training on historical blockchain data, fostering "AI common sense."

## 2. Agent Task Protocol

**Your primary source for assigned tasks is the `docs/plans/` directory.** Monitor the markdown files within this directory (e.g., `grok_next.md`, `gemini_next.md`) to understand your current objectives and project plans.

## 3. Key Files & Scripts

- **`scanner.py`**: The main tool for steganography detection.
- **`diag.py`**: Verifies the integrity and structure of all datasets in the `datasets/` directory.
- **`datasets/[contributor_submission_year]/`**: The core directory structure for all contributions. Each contains `clean/` and `stego/` images, `data_generator.py`, and a `model/` directory.
- **`docs/plans/*.md`**: **Source of truth for agent tasks.**

## 4. Core Development Workflow
All commands are run from the project root unless specified.

1.  **Generate Datasets**:
    ```bash
    cd datasets/<contributor_name>
    python3 data_generator.py --limit 10
    ```

2.  **Verify Data Integrity**:
    ```bash
    python3 diag.py
    ```

3.  **Train Model**:
    Ask user to run the train command is preferred because training duration exceeds most command timeout limit.
    ```bash
    python3 trainer.py
    ```
    *This saves `detector.onnx` and other model files to the `models/` subdirectory.*

4.  **Run Detection**:
    ```bash
    # Scan a single file with full details
    python3 scanner.py /path/to/image.png --json

    # Scan a directory quickly
    python3 scanner.py /path/to/images/ --workers 4
    ```

## 5. Contribution Structure

### Dataset Contribution (`DATASET_GUIDELINES.md`)

- **Location**: `datasets/[username]_submission_[year]/`
- **Structure**: Must contain `clean/` and `stego/` directories with matching filenames.
- **Generation**: Can be done by providing images directly or via a `data_generator.py` script that processes seed files (e.g., `sample_seed.md`).
- **Naming**: `{payload_name}_{algorithm}_{index}.{ext}` (e.g., `seed1_alpha_001.png`).

## 5. Communication with another AI

Use markdown document to communicate with other AIs.
Write your communication to other AI in docs/coordination.

Alternatively, if you like to have fun, you can try
Alpha protocol to communicate in docs/coordination/[username]-[date].png

To read:

```bash
  ./scripts/starlight_extractor.py docs/coordination/[username]-[date].png
```

To write:

```bash
  ./scripts/stego_tool.py embed --input datasets/sample_submission_2025/clean/clean-0039.png --output docs/coordination/[username]-[date].png --method alpha --message "your message"
```

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

<!-- bv-agent-instructions-v1 -->

---

## Beads Workflow Integration

This project uses [beads_viewer](https://github.com/Dicklesworthstone/beads_viewer) for issue tracking. Issues are stored in `.beads/` and tracked in git.

### Essential Commands

```bash
# View issues (launches TUI - avoid in automated sessions)
bv

# CLI commands for agents (use these instead)
bd ready              # Show issues ready to work (no blockers)
bd list --status=open # All open issues
bd show <id>          # Full issue details with dependencies
bd create --title="..." --type=task --priority=2
bd update <id> --status=in_progress
bd close <id> --reason="Completed"
bd close <id1> <id2>  # Close multiple issues at once
bd sync               # Commit and push changes
```

### Workflow Pattern

1. **Start**: Run `bd ready` to find actionable work
2. **Claim**: Use `bd update <id> --status=in_progress`
3. **Work**: Implement the task
4. **Complete**: Use `bd close <id>`
5. **Sync**: Always run `bd sync` at session end

### Key Concepts

- **Dependencies**: Issues can block other issues. `bd ready` shows only unblocked work.
- **Priority**: P0=critical, P1=high, P2=medium, P3=low, P4=backlog (use numbers, not words)
- **Types**: task, bug, feature, epic, question, docs
- **Blocking**: `bd dep add <issue> <depends-on>` to add dependencies

### Session Protocol

**Before ending any session, run this checklist:**

```bash
git status              # Check what changed
git add <files>         # Stage code changes
bd sync                 # Commit beads changes
git commit -m "..."     # Commit code
bd sync                 # Commit any new beads changes
git push                # Push to remote
```

### Best Practices

- Check `bd ready` at session start to find available work
- Update status as you work (in_progress → closed)
- Create new issues with `bd create` when you discover tasks
- Use descriptive titles and set appropriate priority/type
- Always `bd sync` before ending session

<!-- end-bv-agent-instructions -->
