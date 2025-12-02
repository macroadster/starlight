# GPT-OSS Weekly Plan – Starlight ↔ Stargate Production Push
**Updated:** December 1, 2025  
**Phase:** 3 (Production Readiness)  
**Timeline:** Week of Dec 1-6, 2025  
**Agent:** ChatGPT / GPT-OSS (Deployment, API, Stargate integration)  
**Status:** Build-first; ship working clusters and REST contracts

---

## 🎯 Strategic Context
- Starlight now exposes REST-only inscribe flow that hands images to Stargate; Stargate runs on Postgres/JSONB with Helm-managed stack.
- Current focus: harden ingestion, storage, and UI so the shared Kubernetes deployment is usable without filesystem tricks.
- Keep scope on verifiable production behavior (Helm + Docker + Postgres) and lock the REST contracts used by Stargate.

---

## 📋 Objectives This Week

### 1) REST Contracts & Inscription Workflow
- Finalize and publish REST API spec covering `/health`, `/scan/block`, `/scan/file`, `/inscribe` (text→image only), and ingestion to Stargate.
- Ensure Starlight writes pending inscriptions to disk and optionally POSTs to Stargate ingest; document the on-disk contract.
- Add watcher/ops notes so Stargate can pick up pending inscriptions without shared filesystems.

### 2) Stargate Backend Production Hardening
- Keep Postgres JSONB storage as the source of truth; remove reliance on symlinked block dirs.
- Solidify block extraction logic (ordering, dedupe, pending block handling) and expose ingestion endpoint used by Starlight.
- Validate data paths with Helm (PVCs where needed) and provide rollback/runbook steps.

### 3) Frontend & UX Stability
- Fix scrolling regressions; deliver horizontal infinite scroll without arrows and keep vertical scroll stable.
- Ensure block ordering shows genesis and pending blocks consistently; surface text inscription previews when available.
- Ship rebuilt frontend image and redeploy via Helm.

### 4) Deployment & Verification
- Maintain Helm chart in `starlight-helm/` covering backend, frontend, starlight API, and Postgres with PVCs.
- Build and publish fresh Docker images; perform `helm upgrade` and verify cluster health via `kubectl` + port-forward.
- Document access paths (ports, hostnames) and any config required for operators.

### 5) Ops Notes & Risks
- Capture known limitations (no data migration needed yet, Postgres text-read support gaps) and mitigation steps.
- Outline next actions if ingestion or block extraction fails in prod (retry/backfill strategy, alerting hooks).

---

## 🗓️ Day-by-Day
| Day | Focus | Exit Criteria |
| --- | ----- | ------------- |
| Mon | REST spec + inscribe flow | Spec updated; on-disk/ingest contracts documented |
| Tue | Backend hardening | JSONB flow validated; pending block logic cleaned; Helm values align |
| Wed | Frontend polish | Scroll behavior fixed; infinite horizontal scroll working |
| Thu | Images + Helm deploy | Fresh images built; `helm upgrade` succeeds; pods healthy |
| Fri | Ops docs | Access instructions, rollback steps, and known risks captured |

---

## 🔄 Coordination
- Stargate remains the broadcaster; Starlight stops at image inscription/persist + optional REST handoff.
- Prefer REST over filesystem sharing; any temporary glue documented with cleanup path.
- Keep Helm/values in sync across repos; note any manual port-forwards required for testing.

---

**Next Review:** End of week (Dec 6, 2025).**
