# BeatAhead Cardiac Physiological ML Prototype: Dataset Expansion Plan v2

**Document:** `reports/dataset_expansion_plan_v2.md`  
**Target:** 100-Case Dataset Expansion & Machine Learning Preparation  
**Date:** September 2026  
**Status:** PROPOSED PLAN — STOPPED AWAITING HUMAN REVIEW BEFORE DOWNLOAD  

---

## 1. Executive Summary & Review of Pilot Label Audit

A rigorous label audit of the 20-case VitalDB pilot cohort (`reports/corrected_label_audit.md`) identified a critical discrepancy in the initial label generator:
* **The Defect**: The initial code evaluated `(valid_st_win <= -1.0).sum() >= 60`, which erroneously accumulated non-consecutive, fragmented dips (e.g. 8s, 10s, 16s, 34s) to flag 11 false-positive sustained windows in Case 1.
* **The Correction**: The algorithm was refactored to require a true contiguous run length:
  $$\text{max\_consecutive\_duration}(\text{ST\_II} \le -1.0\text{ mm}) \ge 60\text{ seconds}$$
  accounting for the 2-second monitor sampling period and treating missing observations $> 2\text{ seconds}$ as run-terminating dropouts.
* **Audit Results on 20 Pilot Cases**:
  * Usable 5-minute windows: **3,486 windows**
  * Corrected sustained positive windows: **6 windows (0.17%)** (down from 17)
  * Unique positive sustained episodes: **EXACTLY 1 UNIQUE EPISODE** (Case 1, 186 consecutive seconds, nadir: $-1.9\text{ mm}$)
  * Number of cases containing sustained episodes: **1 case (Case 1, 5% of pilot cohort)**

### The Strategic Imperative for Dataset Expansion
In an unselected general surgical population, true sustained ischemic repolarization changes ($\ge 60\text{s}$) occur in only ~5% of patients. With only 1 positive case in the 20-case pilot, **a subject-level train/validation/test split is mathematically impossible** (one partition would hold the single positive case while the other two partitions would have zero positive events).

Therefore, this expansion plan scales the dataset to **100 cases**, introducing a **two-cohort sampling strategy** that balances the necessity of acquiring enough positive events to train models with the need to benchmark on an unbiased representative population.

---

## 2. Two-Cohort Dataset Expansion Strategy ($N = 100\text{ Cases}$)

> [!WARNING]
> **Outcome-Enrichment Bias Warning**:
> Selecting all 100 cases using an outcome filter (e.g. `min(ST_II) <= -1.0 mm`) would introduce massive selection bias, rendering real-world false-alarm rates, negative predictive values, and specificity metrics completely untrustworthy.
> Conversely, sampling 100 cases completely at random would yield only ~3–5 positive cases, leaving validation and test partitions severely underpowered.

To resolve this trade-off, the 100-case expansion is split into two distinct, explicitly labeled cohorts:

```
+----------------------------------------------------------------------------------------------------+
|                                  100-CASE EXPANSION COMPOSITION                                    |
+------------------------------------+----------------------------------+----------------------------+
| Cohort Name                        | Enriched Development Cohort (A)  | Representative Cohort (B)  |
+------------------------------------+----------------------------------+----------------------------+
| Target Sample Size                 | 50 Cases                         | 50 Cases                   |
| Primary Objective                  | Model Development & Debugging:   | Real-World Evaluation:     |
|                                    | Acquire sufficient true positive | Unbiased assessment of     |
|                                    | ischemic episodes                | specificity & false alarms |
+------------------------------------+----------------------------------+----------------------------+
| Selection Methodology              | Pre-event Clinical Risk Profile: | Stratified Random Sampling |
|                                    | - Preoperative HTN (preop_htn=1) | across departments without |
|                                    | - Preoperative DM (preop_dm=1)   | any filtering on ST values |
|                                    | - Age >= 65 years                | or clinical outcomes.      |
|                                    | - Vascular / CAD diagnosis       | Duration: 1.5 - 4.5 hours. |
|                                    | - 100% coverage of Lead V5       | Matching typical surgery.  |
+------------------------------------+----------------------------------+----------------------------+
| Role in ML Partitions              | Distributed across Train & Val   | Held-out Test Set & Val    |
+------------------------------------+----------------------------------+----------------------------+
| Population Claim                   | ENRICHED HIGH-RISK SUBGROUP      | SURGICAL COHORT SAMPLE     |
|                                    | (Does NOT represent general pop) | (Does NOT represent amb.)  |
+------------------------------------+----------------------------------+----------------------------+
```

### 2.1 Cohort A: Enriched Development Cohort ($N = 50\text{ Cases}$)
* **Candidate Pool**: 851 cases meeting duration criteria (1.5–4.5 hours) with documented cardiovascular comorbidity (`preop_htn = 1`, `preop_dm = 1`, age $\ge 65$, or surgical diagnosis containing CAD, angina, infarction, aneurysm, or bypass).
* **Lead V5 Inclusion**: All 50 selected cases possess simultaneous `SNUADC/ECG_II` and `SNUADC/ECG_V5` (100% V5 coverage).
* **Expected Event Yield**: Estimated 10 to 16 unique sustained ischemic episodes, providing the positive training mass necessary to learn subtle pre-event autonomic and vascular markers.

### 2.2 Cohort B: Representative Evaluation Cohort ($N = 50\text{ Cases}$)
* **Candidate Pool**: 3,906 cases with duration 1.5–4.5 hours and core signals.
* **Selection Protocol**: Uniform random sampling (random seed = 42) stratified across major surgical departments (General surgery, Thoracic surgery, Urology, Gynecology).
* **Strict Blinding**: Selected completely blind to ST deviation values, troponin levels, and intraoperative outcomes.
* **Expected Event Yield**: Estimated 1 to 3 positive cases, providing a realistic ~2–5% prevalence benchmark for specificity and false-alarm auditing.

---

## 3. Required Biosignal Tracks & Lead V5 Expansion

```
+----------------------------------------------------------------------------------------------------+
|                                    REQUIRED TRACK SPECIFICATIONS                                   |
+----------------------+---------------------------+---------------+---------------------------------+
| Modality             | Track Name                | Requirement   | Purpose                         |
+----------------------+---------------------------+---------------+---------------------------------+
| 1. ECG Lead II       | SNUADC/ECG_II             | MANDATORY     | Rhythm, inferior ST deviation,  |
|                      |                           | (100/100)     | HRV time/frequency metrics      |
+----------------------+---------------------------+---------------+---------------------------------+
| 2. ECG Lead V5       | SNUADC/ECG_V5             | WHERE AVAIL.  | Anterior/lateral wall ischemia  |
|                      |                           | (65/100 cases)| (essential clinical coverage)   |
+----------------------+---------------------------+---------------+---------------------------------+
| 3. PPG Waveform      | SNUADC/PLETH              | MANDATORY     | Peripheral pulse morphology,    |
|                      |                           | (100/100)     | PAT delay, dP/dt upslope        |
+----------------------+---------------------------+---------------+---------------------------------+
| 4. SpO2 Numeric      | Solar8000/PLETH_SPO2      | MANDATORY     | Continuous arterial saturation, |
|                      |                           | (100/100)     | desaturation depth & nadir      |
+----------------------+---------------------------+---------------+---------------------------------+
| 5. ST Deviation II   | Solar8000/ST_II           | MANDATORY     | Ground truth target labeling    |
|                      |                           | (100/100)     | (inferior wall)                 |
+----------------------+---------------------------+---------------+---------------------------------+
| 6. ST Deviation V5   | Solar8000/ST_V5           | WHERE AVAIL.  | Ground truth target labeling    |
|                      |                           | (where pres.) | (anterolateral wall)            |
+----------------------+---------------------------+---------------+---------------------------------+
```

---

## 4. Temporal Leakage Prevention Architecture

To train a true early-warning model that forecasts ischemic stress before it manifests, we enforce a strict non-overlapping temporal window schema:

```
+----------------------------------------------------------------------------------------------------+
|                                 TEMPORAL EARLY-WARNING FORECASTING                                 |
+----------------------------------------------------------------------------------------------------+

   [  Observation Window (T_obs)  ] ---- [ Prediction Gap (T_gap) ] ---- [  Target Event Horizon (T_target) ]
   <----------- 300 s ------------>      <----------- 300 s ------->      <------------ 300 s ------------>
   t_start                       t_end                                    t_target_start           t_target_end
   - Extract Features:                    ZERO data extracted              Evaluate ground truth:
     * ECG: HRV, QRS, morphology          to guarantee true                * max_consec(ST <= -1.0) >= 60s
     * PPG: PAT, crest time, SQI          early warning                    (Binary Label: 0 or 1)
     * SpO2: mean, nadir, variance
   ----------------------------------------------------------------------------------------------------> Time (t)
   Window Step Size (Stride) = 60 seconds
```

### Strict Exclusion Criteria for Temporal Leakage Prevention
1. **Zero Event Overlap**: Any observation window whose interval $[t_{\text{start}}, t_{\text{end}}]$ overlaps with an active sustained ST event is **strictly excluded**.
2. **Post-Event Exclusion Buffer (15 Minutes)**:
   * Following the resolution of an ST episode, myocardial tissue undergoes a stunning/recovery phase with lingering repolarization changes and autonomic suppression.
   * All windows occurring within **15 minutes (900 seconds)** following the end of an ST episode are excluded from negative control training to prevent contaminated baseline samples.
3. **Pre-Event Buffer ($T_{\text{gap}} = 5\text{ minutes}$)**:
   * The 5-minute gap guarantees that features do not capture the initial downsloping transition of the ST segment itself.

---

## 5. Subject-Level Splitting & The Positive Event Partition Problem

### 5.1 Strict Subject-Level Grouping
* Splitting is performed strictly by **`caseid` / `subjectid`**.
* All sliding windows belonging to a given patient remain exclusively within one partition:
  * **Train Set**: 70% of subjects (~70 cases)
  * **Validation Set**: 15% of subjects (~15 cases)
  * **Test Set**: 15% of subjects (~15 cases) — **LOCKED** until final evaluation.

### 5.2 The Single-Episode Pilot Split Dilemma (Audit Confirmation)
* In the 20-case pilot, there is only **1 positive case (Case 1)** and **1 unique positive episode**.
* If a 70/15/15 subject-level split were executed on the pilot:
  * Putting Case 1 in Train results in Validation and Test having **0 positive episodes** (evaluation impossible).
  * Putting Case 1 in Test results in Train having **0 positive episodes** (model cannot train).
* **Mathematical Confirmation**: Subject-level splitting is impossible on the 20-case pilot without severe bias or zero-event partitions.

### 5.3 100-Case Partition Allocation Strategy
With the 100-case expansion (Cohort A + Cohort B), positive cases will be stratified at the subject level across folds:
* **Train Set (70 cases)**: ~35 Enriched cases, ~35 Representative cases ($\approx 7–10\text{ positive cases}$, $\approx 10–15\text{ unique episodes}$).
* **Validation Set (15 cases)**: ~8 Enriched cases, ~7 Representative cases ($\approx 2–3\text{ positive cases}$, $\approx 2–4\text{ unique episodes}$).
* **Test Set (15 cases)**: ~7 Enriched cases, ~8 Representative cases ($\approx 2–3\text{ positive cases}$, $\approx 2–4\text{ unique episodes}$) — locked.

---

## 6. Pulse Arrival Time (PAT) Quality & Signal Quality Indexing (SQI)

The pilot review proved that the 61.9% plausibility rate was caused by probe disconnection (clamping at $+25,686.95$) and peripheral vasoconstriction. To handle this without data loss or fabrication:

### 6.1 Do Not Blindly Impute Missing PAT
* Converting a flatlined or disconnected sensor into an imputed normal PAT (e.g. 200 ms) creates synthetic false information.
* If a sensor fails, the model must be informed that the sensor failed.

### 6.2 Explicit PAT Quality Feature Vector
For every 5-minute window, the pipeline computes:
1. `PAT_median_ms`: Median PAT among valid beats (or `NaN` if no valid beats).
2. `PAT_iqr_ms`: Interquartile range of PAT (vascular compliance stability).
3. `PAT_valid`: Binary flag:
   $$\text{PAT\_valid} = 1 \iff (\text{PPG\_var} \ge 0.05) \land (\text{PPG} < 20,000) \land (\ge 30\text{ valid paired beats in window})$$
4. `PPG_perfusion_index`: $\text{AC} / \text{DC}$ pulse amplitude ratio.
5. `PPG_valid_fraction`: Fraction of detected ECG R-peaks that yielded a plausible systolic pulse ($100\text{ ms} \le \text{PAT} \le 450\text{ ms}$).
6. `ECG_beat_count`: Total detected QRS complexes in window.

---

## 7. Feature Leakage Audit Table

Before engineering features for XGBoost, all candidate features are audited and categorized into 5 strict classes:

```
+----------------------------------------------------------------------------------------------------+
|                                    FEATURE LEAKAGE AUDIT TABLE                                     |
+----------------------+--------------------+--------------------------------------------------------+
| Feature Candidate    | Classification     | Rationale & Mitigation Protocol                        |
+----------------------+--------------------+--------------------------------------------------------+
| 1. ST_II in Horizon  | TARGET-DERIVED     | STRICTLY PROHIBITED. Ground truth target itself.       |
|    (T_target)        |                    | Must never be present in feature matrix.               |
+----------------------+--------------------+--------------------------------------------------------+
| 2. Raw ST_II level in| POTENTIAL TEMPORAL | DANGEROUS. If absolute ST level is included, model may |
|    Observation (T_obs| LEAKAGE            | simply predict future depression from current baseline.|
|                      |                    | Mitigation: Use baseline-subtracted deviation (delta-ST|
|                      |                    | relative to surgery start) or exclude raw ST entirely. |
+----------------------+--------------------+--------------------------------------------------------+
| 3. HRV Time-Domain   | SAFE PRE-EVENT     | Measures cardiac autonomic regulation (SDNN, RMSSD,    |
|    (RMSSD, SDNN)     | FEATURE            | pNN50). Derived strictly within T_obs.                 |
+----------------------+--------------------+--------------------------------------------------------+
| 4. HRV Freq-Domain   | SAFE PRE-EVENT     | Autonomic balance marker (VLF, LF, HF, LF/HF ratio).   |
|    (LF, HF, LF/HF)   | FEATURE            | Derived strictly within T_obs.                         |
+----------------------+--------------------+--------------------------------------------------------+
| 5. QRS Morphology    | SAFE PRE-EVENT     | Electrophysiological depolarization marker (duration,  |
|    (width, amp, HF)  | FEATURE            | R-peak amplitude, high-frequency QRS energy).          |
+----------------------+--------------------+--------------------------------------------------------+
| 6. PAT Metrics       | SAFE PRE-EVENT     | Arterial pulse wave transit proxy (PAT median, IQR,    |
|    (median, IQR)     | FEATURE            | pulse crest time). Gated by PAT_valid flag.            |
+----------------------+--------------------+--------------------------------------------------------+
| 7. SpO2 Dynamics     | SAFE PRE-EVENT     | Systemic arterial oxygenation (mean SpO2, nadir,       |
|    (mean, nadir)     | FEATURE            | desaturation area). Derived strictly within T_obs.     |
+----------------------+--------------------+--------------------------------------------------------+
| 8. Invasive Arterial | NOT AVAILABLE IN   | Continuous invasive A-line BP is not present on        |
|    Line MAP (ART_MBP)| REAL-TIME WEARABLE | wearable devices. Prohibited as runtime ML feature.    |
|                      |                    | (Used only as retrospective hemodynamic reference).    |
+----------------------+--------------------+--------------------------------------------------------+
| 9. Vasoactive Drugs  | CONFOUNDER /       | Drug boluses (phenylephrine, ephedrine, epinephrine)   |
|    (bolus & infusion)| CONTROL VARIABLE   | alter vascular tone and PAT independently of ischemia. |
|                      |                    | Retained as clinical covariate for adjustment.         |
+----------------------+--------------------+--------------------------------------------------------+
| 10. Patient Baseline | CONFOUNDER /       | Non-time-varying baseline risk factors (age, sex, BMI, |
|     (Age, HTN, DM)   | CONTROL VARIABLE   | preoperative hypertension/diabetes). Fully safe.       |
+----------------------+--------------------+--------------------------------------------------------+
```

---

## 8. Storage, Bandwidth, & Download Budget

| Asset | Case Count | Unit Size | Total Estimated Size | Storage Location |
| :--- | :--- | :--- | :--- | :--- |
| **Existing Pilot Archive** | 20 cases | 12.2 MB avg | **245 MB** | `data/raw/vitaldb_pilot/` |
| **Cohort A (Enriched)** | 50 cases | 15.5 MB avg | **~775 MB** | `data/raw/vitaldb_100/cohort_a/` |
| **Cohort B (Representative)**| 50 cases | 15.0 MB avg | **~750 MB** | `data/raw/vitaldb_100/cohort_b/` |
| **Total 100-Case Raw .vital**| 100 cases | 15.2 MB avg | **~1.52 GB** | Safe for local disk budget |
| **Extracted Multi-track NPY**| 100 cases | ~10 MB/case | **~1.0 GB** | Resampled 500 Hz waveforms |
| **Final Engineered Matrix**  | 100 cases | ~600 KB/case| **~60 MB** | Windowed tabular CSV/Parquet |

### Exact Case IDs Selected for Next Download Phase (Staged):
* **Cohort A (Enriched, $N=50$)**:
  `[2605, 3614, 4290, 2043, 4247, 2034, 3390, 3526, 758, 1159, 2197, 2772, 4272, 2824, 3037, 3632, 2800, 3192, 4519, 2854, 4038, 2037, 2717, 3027, 4390, 4220, 2697, 4423, 2085, 3505, 3042, 3535, 4252, 2727, 3426, 4236, 4243, 4434, 2725, 3418, 4437, 2081, 2623, 4246, 3530, 4568, 2075, 4435, 4244, 4234]`
* **Cohort B (Representative, $N=50$)**:
  `[1268, 485, 2630, 4498, 3948, 3754, 5799, 3216, 864, 1923, 497, 1855, 3687, 5779, 1374, 1344, 4225, 4016, 5635, 3398, 4771, 381, 2866, 1714, 2419, 4467, 360, 481, 3486, 2728, 5585, 2172, 6081, 4616, 2253, 569, 111, 2719, 4349, 1269, 5664, 5326, 3280, 2146, 5130, 3450, 4120, 5262, 2379, 451]`

---

## 9. Next Steps & Hold Point

> [!NOTE]
> **Execution Halted**:
> - NO additional cases have been downloaded.
> - NO machine learning models (XGBoost, Random Forest, Logistic Regression) have been trained.
> - NO synthetic data has been generated.
> - NO ISI has been calculated.
> - The BeatAhead website repository has NOT been modified.
> 
> Awaiting human review and explicit approval of this expansion plan before initiating any downloads.
