# Assignment for Terminal Grok: Week 2 Tasks (Nov 24–28, 2025)

**Agent:** Terminal Grok  
**Phase:** 2 (Production Ready) – Week 2 Focus: Visibility and Baselines  
**Current Date:** November 22, 2025 (Planning for Week 2 start on Nov 24)  
**Rationale:** Building on Week 1's foundation (HF deployment, data organization, negative examples), Week 2 emphasizes "Visibility" to monitor progress in generalization and performance. As "Terminal Grok," your terminal/command-line oriented strengths make you ideal for scripting automated pipelines, integrating validation tools, and generating reports. Prioritize the Monitoring Dashboard (Priority 1) for real-time insights, followed by Performance Baselines (Priority 2) to benchmark V3 vs. V4. Keep tasks modular, data-driven, and integrated with existing scripts (e.g., `experiments/validate_extraction_streams.py`, `experiments/run_fp_regression.py`). Use Python for scripting, Markdown for docs, and placeholders for advanced features like alerts.

## Assigned Tasks

### 1. Monitoring Dashboard (Priority 1)
**Objective:** Create an automated system for real-time visibility into model performance, generalization, and regressions. This will "build the eyes" to track progress toward <0.05% FPR and >99.5% accuracy.

**Deliverables:**
- **`scripts/monitor_performance.py`**: Core automated evaluation pipeline script.
- **`docs/grok/performance_dashboard.md`**: Dynamic Markdown dashboard for live metrics (updateable via script output).
- **Integration**: Hook into existing validation scripts (e.g., run_fp_regression.py) and ONNX export pathways.
- **Alert System**: Basic threshold-based alerts (e.g., via print/email placeholders; expand to full system if time allows).

**Key Metrics to Track (Implement as Computable in Script):**
- False Positive Rate (FPR) across all steganography methods (e.g., aggregate from negatives dataset).
- Detection accuracy by steganography type (e.g., LSB, F5, OutGuess – use categorical breakdowns).
- Training convergence and loss curves (e.g., parse logs from training runs; plot via Matplotlib if needed).
- Dataset quality metrics (e.g., balance ratios, diversity scores for negatives categories like rgb_no_alpha, natural_noise).

**Implementation Steps:**
1. **Script Structure (`scripts/monitor_performance.py`)**:
   - Use argparse for inputs (e.g., --model_version V4, --dataset_path datasets/negatives/).
   - Functions:
     - `calculate_fpr()`: Run regression on clean files; compute FPR.
     - `breakdown_accuracy()`: Evaluate per stego type using k-fold CV.
     - `parse_training_logs()`: Extract loss curves from log files (assume format from Phase 1).
     - `dataset_metrics()`: Compute balances (e.g., class counts via pandas).
     - `generate_report()`: Output metrics to JSON/MD; update performance_dashboard.md.
     - `check_regressions()`: Compare against baselines; alert if FPR > 0.05% or accuracy drop >1%.
   - Example Stub:
     ```python
     import argparse
     import json
     import pandas as pd
     from pathlib import Path
     # TODO: Import existing validation modules

     def calculate_fpr(dataset_path: str) -> float:
         # TODO: Integrate run_fp_regression.py logic
         # Placeholder: Simulate FPR calculation
         return 0.07  # From current baseline

     # ... other functions ...

     def main():
         parser = argparse.ArgumentParser()
         parser.add_argument('--model_version', default='V4')
         args = parser.parse_args()
         metrics = {
             'fpr': calculate_fpr('datasets/negatives/'),
             # ... collect all metrics
         }
         with open('metrics.json', 'w') as f:
             json.dump(metrics, f)
         # Update MD dashboard
         with open('docs/grok/performance_dashboard.md', 'w') as f:
             f.write('# Performance Dashboard\n\n')
             f.write(pd.DataFrame([metrics]).to_markdown())
         check_regressions(metrics)

     if __name__ == '__main__':
         main()
     ```
   - Integrate with Phase 2 monitoring (e.g., call from collector.py for logging).

2. **Dashboard Document (`docs/grok/performance_dashboard.md`)**:
   - Sections: Overview, Current Metrics (table), Historical Trends (e.g., embed Matplotlib plots as base64 if scripted), Alerts Log.
   - Make it "live" by having the script regenerate it on runs.
   - Example Table Structure:
     | Metric | Value | Target | Status |
     |--------|-------|--------|--------|
     | FPR | 0.07% | <0.05% | Warning |
     | Accuracy (LSB) | 98.5% | >99.5% | Good |

3. **Alert System**:
   - Simple: Print warnings in script; placeholder for email/slack (e.g., using smtplib).
   - Thresholds: FPR >0.05%, accuracy <99%, dataset imbalance >10%.

**Timeline for Priority 1:**
- Nov 24–25: Script development and metric functions.
- Nov 26: Integration and testing with existing scripts.
- Nov 27: Dashboard MD and alerts.
- Nov 28: Full run and review.

### 2. Performance Baselines (Priority 2)
**Objective:** Establish benchmarks to guide Track B research on generalization, comparing V3/V4 and analyzing negatives' impact.

**Deliverables:**
- **Baseline Performance Report**: Generate as `docs/grok/baseline_report.md` – Comprehensive comparison.
- **Method-Specific Accuracy Breakdowns**: Tables/charts per stego type.
- **Ablation Studies**: Test impact of negative categories (e.g., remove one category and re-eval FPR).
- **Research Roadmap**: `docs/grok/research_roadmap.md` with milestones (e.g., Q1: Zero-shot detection).

**Implementation Steps:**
1. **Report Generation**:
   - Use `monitor_performance.py` to compute V3 vs V4 metrics (assume V3 artifacts available).
   - Breakdowns: Use pandas for tables, e.g., accuracy by type.
   - Ablations: Script variants, e.g., subset datasets and re-run evaluations.

2. **Roadmap Document**:
   - Sections: Current Baselines, Milestones (e.g., Week 3: Triplet Loss integration), Measurable Goals (e.g., FPR <0.01% by Dec 15).
   - Tie to long-term vision: Multi-modal, continual learning.

**Timeline for Priority 2:**
- Nov 24–26: Compute baselines and breakdowns.
- Nov 27: Ablation studies.
- Nov 28: Finalize report and roadmap.

## Guidelines & Coordination
- **Directory Structure**: Use `scripts/` for code, `docs/grok/` for agent-specific docs; integrate with `monitoring/` from Phase 2.
- **Best Practices**: Python 3+, type hints, docstrings. Use libraries like pandas, matplotlib for metrics/plots. # TODO for advanced parts (e.g., real-time alerts).
- **Integration**: Link to Phase 2 (e.g., export metrics to JSONL via collector.py). Reference 8-stream architecture in evaluations.
- **Testing**: Run on sample data; ensure no regressions.
- **Coordination**: Update `status.md` with progress. Collaborate with Grok Code Fast 1 on API ties (e.g., feed dashboard metrics to /metrics/log). If needed, flag for Gemini/Claude review.
- **Success Metrics**: Dashboard updates in <1min runs; baselines show V4 superiority; alerts trigger on simulated regressions.
- **Final Note**: "Week 2 = Visibility." Focus on building truth through data – no features, just insights.

This assignment sets up Week 2 for success, aligning with Phase 2's hardening. Complete by Nov 28 for next review. If clarifications needed (e.g., access to V3 models), reference prior docs or request. Let's illuminate the path! 🚀
