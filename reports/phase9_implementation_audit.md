# Phase 9 — Documentation Consistency Audit Report
**Document ID:** BEATAHEAD-PHASE9-AUDIT-THRESHOLD  
**Execution Date:** 2026-09-22 20:15:00 UTC  
**Scope:** Verification of Decision Threshold Derivation & Optimization Metric  
**Status:** COMPLETE (Zero Threshold Modification / Zero Retraining)  

---

## 1. Audit Objective

Task 0 mandates an inspection of the Phase 5 and Phase 6 benchmarks and reporting artifacts to determine whether the frozen decision threshold **`0.156742`** was selected using:
- **$F_1$-score maximization** (`argmax(F1)`), or
- **$F_2$-score maximization** (`argmax(F2)`).

The objective is to eliminate documentation discrepancies and ensure that all records consistently and accurately describe the exact mathematical method used to calibrate the already-frozen threshold.

---

## 2. Evidence from Primary Artifacts & Code

We conducted an exhaustive audit across the codebase, configuration artifacts, and serialized metadata:

### 2.1 Primary Training Script (`src/train_and_evaluate_models.py`)
In the primary cross-validation and benchmarking pipeline:
- Lines 321–324:
  ```python
  f1_scores = 2 * (prec * rec) / np.maximum(prec + rec, 1e-8)
  best_f1_idx = np.argmax(f1_scores)
  thresh_best_f1 = thresholds[min(best_f1_idx, len(thresholds)-1)] if len(thresholds) > 0 else 0.5
  best_f1 = f1_scores[best_f1_idx]
  ```
- Lines 386–390:
  ```python
  # Use thresholds calibrated on Dev out-of-fold predictions
  oof_probs = cv_results[m_name]["oof_preds"]
  thresh_info = calibrate_thresholds(y_dev, oof_probs)
  thresh = thresh_info["thresh_max_f1"]
  ```
- Line 428: The calibrated threshold recorded into `reports/phase5_model_benchmark_results.csv` was `thresh_max_f1`.
- **Finding**: The training and benchmarking script explicitly calibrated the threshold using **$F_1$-score maximization** (`argmax(F1)`).

### 2.2 Serialized Model Metadata (`models/model_metadata.json`)
In the immutable Phase 6 deployment metadata:
- Lines 147–150:
  ```json
  "calibration_and_decision_threshold": {
    "deployment_decision_threshold": 0.15674158930778503,
    "threshold_calibration_strategy": "Out-of-fold cross-validation F1-score maximization (argmax(F1))",
    "threshold_calibrated_on": "Development Set (80 cases) 5-fold cross-validation exclusively"
  }
  ```
- **Finding**: Model metadata explicitly documents **$F_1$-score maximization (`argmax(F1)`)**.

### 2.3 Deployment Readiness Report (`reports/phase6_deployment_readiness.md`)
- Table on Line 36:
  `| Deployment Decision Threshold | 0.156742 (calibrated via Dev out-of-fold F1 maximization) |`
- **Finding**: Phase 6 deployment readiness report explicitly documents **$F_1$-score maximization**.

---

## 3. Discrepancy Identification & Resolution

### 3.1 Discrepancy Identified
In `reports/phase8_1_isi_sensitivity_audit.md` and associated summary text, Section 3 ("Parameter Summary Table") and Section 9 inadvertently included the phrase:
- *"Optimal F2 utility threshold from Phase 5 benchmark"*

### 3.2 Resolution & Clarification
1. The mathematical code in `src/train_and_evaluate_models.py`, `scratch/build_deployment_artifacts.py`, and `models/model_metadata.json` never implemented $F_2$ optimization; the threshold has always been **$0.15674158930778503$ (rounded to $0.156742$)**, derived strictly from **out-of-fold $F_1$-score maximization (`argmax(F1)`)**.
2. The reference to "$F_2$" in the Phase 8.1 narrative was an editorial typo referencing a preliminary clinical discussion on recall weighting, but the actual deployed threshold was and remains the **$F_1$-calibrated threshold**.
3. In accordance with Task 0 instructions, **no threshold was recalculated or modified**. The value remains frozen at **`0.156742`**.
4. All documentation is now aligned: **Threshold `0.156742` was calibrated via out-of-fold $F_1$-score maximization on the 80-case development set.**
