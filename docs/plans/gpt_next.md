# GPT-OSS Phase 2 Tasks
**Version:** 1.0  
**Phase:** 2  
**Scope:** Documentation + Monitoring Backend (Safe Tasks Only)

---

## 1. Documentation Tasks (Primary)

### 1.1 Update `V4_UTILS_SPEC.md`
- Sync argument names, shapes, and stream descriptions.
- Ensure PNG/JPEG differences are accurately documented.
- Add section describing 8-stream deterministic shapes.
- Add notes on preprocessing invariants required for ONNX export.

### 1.2 Create `PHASE_2_DOCS.md`
Contents:
- Overview of V4 architecture.
- Phase 2 deployment flow (export → quantize → containerize).
- Monitoring and logging overview.
- Summary of 8 streams and purpose of each.
- Preprocessing pipeline diagram in markdown or ASCII.

### 1.3 Create monitoring documentation
Files:
- `MONITORING_SPEC.md`
- `METRICS_REFERENCE.md`
- `LOGGING_SCHEMA.md`

Details to include:
- List of metrics (latency, throughput, fp/fn counters, format distribution, drift indicators).
- JSON schema for logging.
- Example log entries.
- Directory structure for monitoring module.

### 1.4 Developer docs
- `DEVELOPER_GUIDE.md`
- Instructions:
  - Run preprocess-only.
  - Run tests.
  - Run scanner against a directory.
  - Troubleshooting common issues.

---

## 2. Monitoring Backend Scaffolding (Secondary)

### 2.1 Logging schema
Create Python file: `monitoring/schema.py`
- Define `METRIC_SCHEMA` as a Python dict.
- Include fields: timestamp, latency_ms, throughput, image_format, model_version, fp_flag, fn_flag.

### 2.2 Simple logging utilities
Create: `monitoring/collector.py`
- Functions:
  - `log_latency(ms: float) -> None`
  - `log_metric(metric_dict: dict) -> None`
  - `save_jsonl(path: str) -> None`
- Function bodies remain TODO placeholders.

### 2.3 Monitoring API stubs
Create: `api/metrics.py`
- FastAPI route stubs only:
```python
@router.post("/metrics/log")
def log_metric(payload: dict):
    return {"status": "ok"}

