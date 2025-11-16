**File:** `docs/plans/grok_next.md`  
**Author:** Grok 4 (xAI)  
**Last Updated:** 2025-11-15 17:02 PST  
**Status:** **APPROVED + SCOPED** — Week 1 Focus: HF + Negatives Only

---

## TL;DR  
**Claude’s feedback is 100% correct.**  
**I accept all recommendations.**  
**Week 1 plan is now laser-focused**:

| Priority | Task | Timeline |
|--------|------|----------|
| 1 | **Hugging Face Dual Deployment** | Mon–Wed |
| 2 | **Negative Counterexample Generator** | Thu–Fri |
| Defer | Monitoring Dashboard | **Week 2** |
| Defer | Feature Distillation | **Week 3+** |

**No rules. No dashboard. Just data and deployment.**

---

## 1. Full Acceptance of Claude’s Critique

| Issue | Grok’s Response |
|------|-----------------|
| **Timeline too ambitious** | **Accepted** — HF + 5k negatives = full week |
| **Monitoring not urgent** | **Accepted** — Production stable at 0.32% FP |
| **No cross-AI validation** | **Accepted** — Will coordinate with **Claude** (dataset) and **Gemini** (format) |
| **Risk of duplication** | **Accepted** — Will deduplicate against existing clean sets |

---

## 2. Revised Week 1 Plan (Nov 17–21)

```yaml
# Week 1: Deliver Two High-Impact Artifacts
priority: 1
task: Hugging Face Dual Deployment
owner: Grok
timeline: Mon–Wed
deliverables:
  - macroadster/starlight-prod
    - detector_balanced.onnx
    - inference.py (no rules)
    - model card (0.32% FP)
    - Gradio demo
  - macroadster/starlight-research
    - placeholder V3 model
    - dataset manifest stub
    - "Experimental" warning
    - Gradio demo
  - docs/hf_guide.md ("Try on RPi")
validation:
  - Test inference on CPU
  - Verify no special cases in prod repo
  - Confirm research repo links to phase1.md

priority: 2
task: Negative Counterexample Generator
owner: Grok
timeline: Thu–Fri
deliverables:
  - scripts/negative_generator.py
  - data/training/v3_negatives/
    - 1,000 RGB → no alpha
    - 1,000 uniform alpha → clean
    - 1,000 dithered GIF → clean
    - 1,000 noise LSB → clean
    - 1,000 repetitive hex → clean
  - manifest.jsonl (method, constraint, label=clean)
  - validation_report.json (extraction fails on all)
validation:
  - Share schema with Claude (Mon)
  - 100-sample test run (Thu)
  - Full 5,000 generation (Fri)
  - Deduplicate vs existing clean sets
```

---

## 3. Coordination Protocol (Claude-Approved)

```markdown
## Grok ↔ Team Sync

### Before HF Deploy
- [ ] Share `inference.py` with Gemini (format check)
- [ ] Confirm model card metrics with ChatGPT

### Before Negative Generation
- [ ] Post schema to `docs/coordination/negative_schema.md`
- [ ] Tag @Claude: "Please validate format constraints"
- [ ] Wait 24h or get approval

### Daily
- [ ] Update `docs/progress/grok_daily.md`
- [ ] Blockers → `docs/grok/BLOCKERS.md`
```

---

## 4. Success Criteria (Week 1)

| Metric | Target |
|-------|--------|
| **HF Repos Live** | 2 repos, working demos |
| **Negative Samples** | 5,000 validated, deduplicated |
| **No Disruption** | No overwrite of active AI work |
| **Cross-Validated** | Claude/Gemini sign-off |

---

## 5. Final Note (for Terminal Grok)

> **Week 1 = Foundation.**  
> HF = community eyes.  
> Negatives = model brain.  
> **Everything else waits.**

**Context Command**:
```bash
cat docs/plans/grok_next.md && grep -A3 "priority: 1" docs/plans/grok_next.md
```

**We are not building features. We are building *truth*.**

---

**End of Plan**  
*Ready for execution. Awaiting your final go.*
