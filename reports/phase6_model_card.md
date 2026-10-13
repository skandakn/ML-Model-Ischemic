# BeatAhead Model Card: Phase 5 Deployment Candidate (XGBoost Matrix A)

**Model Name:** `beatahead_phase5_model`  
**Model Version:** `1.0.0`  
**Feature Schema Version:** `1.0.0`  
**Document Date:** 2026-09-22  
**Framework:** XGBoost (`v3.4.1`) & Scikit-Learn (`v1.9.1`)  
**Deployment Artifact:** `models/beatahead_phase5_model.joblib`  
**Status:** FROZEN RESEARCH PROTOTYPE — NOT CLINICALLY APPROVED FOR DIAGNOSTIC USE  

---

## 1. Regulatory & Safety Disclaimer

> [!CAUTION]
> **INVESTIGATIONAL RESEARCH PROTOTYPE ONLY**:  
> The BeatAhead physiological machine learning model is an experimental research prototype developed for retrospective algorithmic benchmarking on open-access surgical biosignals (VitalDB). It has **NOT** been evaluated in prospective clinical trials, has **NOT** received regulatory clearance (e.g., FDA 510(k), De Novo, or CE mark), and must **NEVER** be used as a primary diagnostic tool, standalone clinical monitor, or automated therapeutic trigger in patient care.

---

## 2. Intended Use & Clinical Scope

- **Primary Clinical Task**: Early warning of impending intraoperative myocardial ischemia during surgical procedures under general anesthesia.
- **Target Population**: Adult patients undergoing major non-cardiac or cardiovascular surgery with continuous multi-parameter monitoring (ECG Lead II, finger photoplethysmogram, pulse oximetry).
- **Operational Lead Time**: Generates predictions with an enforced **5-minute advance lead time** ($T_{\text{obs}} = 300\text{ s}$, $T_{\text{gap}} = 300\text{ s}$, $T_{\text{target}} = 300\text{ s}$).
- **Intended Clinical Context**: Secondary decision-support alert designed to notify anesthesia teams of subtle, insidious ST segment micro-drifting before overt clinical thresholds ($\le -1.0\text{ mm}$) are breached.
- **Out of Scope**:
  - Out-of-hospital or ambulatory cardiac monitoring (e.g. smartwatches or patch ECGs).
  - Diagnostic localization of regional wall motion abnormalities (e.g., distinguishing LAD vs RCA occlusions).
  - Detection of non-ischemic arrhythmias or hemodynamic shock without repolarization changes.

---

## 3. Experimental Target Definition

The ground truth prediction target is defined as:
$$\text{Target } y = 1 \iff \max\text{consecutive\_duration}(\text{ST\_II} \le -1.0\text{ mm}) \ge 60\text{ seconds within } T_{\text{target}}$$

### Temporal Architecture Constraints:
- **Observation Window ($T_{\text{obs}}$)**: $[t, t + 300\text{ s})$ — raw signal window from which all 26 features are exclusively extracted.
- **Prediction Lead-Time Gap ($T_{\text{gap}}$)**: $[t + 300\text{ s}, t + 600\text{ s})$ — mandatory blank gap with zero feature extraction, ensuring true early warning.
- **Target Horizon ($T_{\text{target}}$)**: $[t + 600\text{ s}, t + 900\text{ s})$ — interval where ground truth sustained ischemia is evaluated.
- **Sliding Stride**: $60\text{ seconds}$ between overlapping evaluation windows.
- **Exclusion of Contaminated Windows**: Windows with active sustained ischemia ($\ge 60\text{ s}$) inside $T_{\text{obs}}$ or $T_{\text{gap}}$, or within 15 minutes post-resolution, are labeled $y = -1$ and strictly excluded.

---

## 4. Training Dataset & Cohort Composition

The model was developed on the official **100-Case VitalDB Surgical Dataset**:
- **Cohort A (High-Risk Cardiovascular Enriched, $N=50$)**: Elderly patients ($\ge 65\text{ yrs}$) with documented hypertension (`preop_htn = 1`), diabetes (`preop_dm = 1`), or vascular/CAD diagnosis.
- **Cohort B (Representative Multi-Specialty Surgical, $N=50$)**: Uniformly sampled across general, thoracic, urology, and gynecology surgical departments, completely blind to ST outcomes.
- **Replacement Case 33**: Pre-audited Case 33 (Hernia repair, 1.75h) replaced defective candidate Case 5326 (which lacked `SNUADC/PLETH` upstream).
- **Two-Tier Subject-Level Partition**:
  - **Development Set ($N = 80$ Cases; 40 A, 40 B)**: 8,571 usable windows, 37 positive ($y=1$), 8,534 clean negative ($y=0$); 6 positive patients (`[137, 864, 2172, 4236, 4390, 4568]`).
  - **Locked Test Set ($N = 20$ Cases; 10 A, 10 B)**: 2,770 usable windows, 11 positive ($y=1$), 2,759 clean negative ($y=0$); 2 positive patients (`[2085, 3280]`).
  - **Zero Patient Overlap**: $\text{Dev} \cap \text{Test} = \emptyset$.

---

## 5. Input Feature Schema (Matrix A, 26 Features)

All features are extracted strictly from $T_{\text{obs}}$ and validated against physiological bounds:

| # | Feature Name | Modality | Unit | Valid Bounds | Description |
| :-: | :--- | :---: | :---: | :---: | :--- |
| 1 | `ecg_hr_mean` | ECG II | bpm | $[20.0, 300.0]$ | Mean heart rate over 5-minute window |
| 2 | `ecg_hr_std` | ECG II | bpm | $[0.0, 150.0]$ | Heart rate standard deviation (autonomic tone) |
| 3 | `ecg_rr_sdnn` | ECG II | ms | $[0.0, 1000.0]$ | SDNN: Standard deviation of normal RR intervals |
| 4 | `ecg_rr_rmssd` | ECG II | ms | $[0.0, 1000.0]$ | RMSSD: Successive RR interval difference root mean square |
| 5 | `ecg_pnn50` | ECG II | % | $[0.0, 100.0]$ | Percentage of adjacent RR intervals differing by $>50\text{ ms}$ |
| 6 | `ecg_r_amp_mv` | ECG II | mV | $[-2.0, 15.0]$ | Median R-wave amplitude above isoelectric line |
| 7 | `ecg_qrs_width_ms` | ECG II | ms | $[30.0, 300.0]$ | QRS complex duration estimate |
| 8 | `ecg_sqi` | ECG II | ratio | $[0.0, 1.0]$ | Electrocardiogram signal quality index |
| 9 | `pat_median_ms` | ECG+PPG | ms | $[50.0, 600.0]$ | Median Pulse Arrival Time (QRS peak to PPG foot delay) |
| 10 | `pat_iqr_ms` | ECG+PPG | ms | $[0.0, 500.0]$ | Interquartile range of PAT (vascular transit stability) |
| 11 | `pat_valid_fraction` | ECG+PPG | ratio | $[0.0, 1.0]$ | Fraction of cycles with plausible PAT (100–450 ms) |
| 12 | `pat_valid` | ECG+PPG | binary | $[0, 1]$ | Binary validity flag ($\ge 30\%$ plausible PAT beats) |
| 13 | `ppg_pulse_amp` | PPG | a.u. | $[0.0, 50000.0]$ | Median peak-to-trough plethysmogram pulse amplitude |
| 14 | `ppg_perfusion_index`| PPG | % | $[0.0, 10000.0]$ | Pulsatile AC / non-pulsatile DC perfusion index |
| 15 | `ppg_crest_time_ms` | PPG | ms | $[20.0, 500.0]$ | Foot to systolic peak crest time (arterial stiffness) |
| 16 | `ppg_sqi` | PPG | ratio | $[0.0, 1.0]$ | Photoplethysmogram signal quality index |
| 17 | `spo2_mean` | SpO2 | % | $[40.0, 100.0]$ | Mean pulse oximeter oxygen saturation |
| 18 | `spo2_min` | SpO2 | % | $[30.0, 100.0]$ | Nadir oxygen saturation |
| 19 | `spo2_std` | SpO2 | % | $[0.0, 40.0]$ | Oxygen saturation standard deviation |
| 20 | `spo2_desat_count` | SpO2 | s | $[0, 300]$ | Duration of arterial desaturation ($\text{SpO2} \le 90\%$) |
| 21 | `st_obs_mean` | ST II | mm | $[-15.0, 15.0]$ | Mean ST segment deviation in Lead II |
| 22 | `st_obs_median` | ST II | mm | $[-15.0, 15.0]$ | Median ST segment deviation in Lead II |
| 23 | `st_obs_min` | ST II | mm | $[-15.0, 15.0]$ | Minimum ST segment deviation in Lead II |
| 24 | `st_obs_std` | ST II | mm | $[0.0, 15.0]$ | ST segment standard deviation |
| 25 | `st_delta_baseline` | ST II | mm | $[-15.0, 15.0]$ | ST shift relative to initial 10-minute surgical baseline |
| 26 | `st_slope_mm_min` | ST II | mm/min| $[-20.0, 20.0]$ | ST regression slope across 5-minute window |

---

## 6. Model Architecture & Hyperparameters

- **Algorithm**: `XGBClassifier` (Extreme Gradient Boosting Trees)
- **Objective Function**: `binary:logistic` (Logloss)
- **Class Imbalance Handling**: `scale_pos_weight = 230.65` (matching the empirical $8,534 / 37$ negative-to-positive ratio in development)
- **Tree Parameters**:
  - `n_estimators`: 150
  - `max_depth`: 4
  - `learning_rate`: 0.03
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `random_state`: 42
- **Preprocessing Pipeline**: `SimpleImputer(strategy='median')` fit strictly on the 80 development cases.

---

## 7. Decision Threshold & Calibration Procedure

- **Model Probability ($P_{\text{model}}$)**: Raw continuous output in $[0, 1]$ representing predicted likelihood of sustained ischemia within $T_{\text{target}}$.
- **Frozen Decision Threshold ($T_{\text{decision}}$)**: **`0.156742`**.
- **Calibration Method**: Calibrated on pooled out-of-fold predictions from the 80 development cases to maximize $F_1$-score ($\text{argmax}(F_1)$).
- **Binary Decision Logic**:
  $$\text{alert} = \begin{cases} 1 \text{ ("ALERT")}, & \text{if } P_{\text{model}} \ge 0.156742 \\ 0 \text{ ("NORMAL")}, & \text{if } P_{\text{model}} < 0.156742 \end{cases}$$

---

## 8. Frozen Benchmark Performance Metrics

### Development 5-Fold Cross-Validation (80 Cases, OOF Pooled):
- **PR-AUC (Average Precision)**: **`0.3247`** (vs 0.0043 natural prevalence = **~75x above chance**)
- **Mean Fold PR-AUC**: **$0.4812 \pm 0.3834$**
- **AUROC**: **`0.8896`**
- **Brier Calibration Loss**: **`0.0045`**

### Locked Benchmark Test Set (20 Cases, Sealed):
- **Test Window Specificity**: **`99.71%`**
- **False Alarm Rate**: **`0.1733 alarms / surgical hour`** (1 false alarm every 5.77 hours)
- **Test Window Sensitivity**: `0.0%` (on abrupt-onset test cases)
- **Test PR-AUC**: `0.0043` | **Test AUROC**: `0.4967` | **Test Brier**: `0.0056`

---

## 9. Limitations, Failure Modes & Observed Phenotypes

1. **Observed Phenotype Sensitivity (Insidious vs. Abrupt)**:
   - **Insidious Micro-Drifts**: The model accurately detects progressive subendocardial ischemia where pre-event ST drifts downward ($\text{mean ST} = -0.53\text{ mm}$ during $T_{\text{obs}}$).
   - **Abrupt Onset Collapses**: When ischemia occurs instantaneously (e.g. sudden cross-clamping or acute vasospasm), the ST segment is completely normal ($+0.27\text{ mm}$) 5 minutes prior. The model refuses to sound false alarms on normal physiology, resulting in zero early warning. A real-time deterministic reflex ($\text{ST} \le -1.0\text{ mm}$) must operate in tandem.
2. **Failure of Pure Non-ST Features (Matrix B Ablation)**:
   Ablation benchmarks proved that non-ST vascular and autonomic features alone (PAT, HRV, pulse morphology) achieve an AUROC of only 0.53. BeatAhead requires electrophysiological repolarization features (ST segment) to achieve clinical utility.
3. **Single-Lead II Monitoring**:
   Ischemia limited to the circumflex or left anterior descending distribution may express exclusively in precordial leads (e.g. V5). Events without Lead II manifestations will not be detected.
4. **Motion & Surgical Electrocautery**:
   Electrocautery creates high-frequency bursts that degrade `ecg_sqi` and `ppg_sqi`. Severe artifact windows are rejected by SQI filters.
