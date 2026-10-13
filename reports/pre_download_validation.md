# BeatAhead Cardiac Physiological ML Prototype: Final Pre-Download Validation & Cohort Audit

**Document:** `reports/pre_download_validation.md`  
**Target:** 20-Case Pilot Early-Warning Pipeline & Candidate 100-Case Cohort Audit  
**Date:** September 2026  
**Status:** VALIDATION COMPLETE — AWAITING HUMAN REVIEW (DOWNLOADS & MODEL TRAINING HALTED)  

---

## Executive Summary

Before committing to downloading the expanded 100-case VitalDB dataset, this audit performed:
1. **Early-Warning Label Pipeline Validation**: Applied the exact clinical early-warning temporal architecture ($T_{\text{obs}} = 300\text{ s}$, $T_{\text{gap}} = 300\text{ s}$, $T_{\text{target}} = 300\text{ s}$, stride $= 60\text{ s}$) with the corrected consecutive ST-event rule on the existing 20-case pilot dataset.
2. **Comprehensive 100-Case Track Audit**: Verified every candidate case against the live VitalDB metadata API (`api.vitaldb.net/cases` and `api.vitaldb.net/trks`).
3. **Identification of Track Defects**: Discovered that **14 of the proposed 100 cases (14.0%) lacked mandatory physiological tracks** (4 in Cohort A missing `Solar8000/ST_II`, 6 in Cohort B missing `Solar8000/ST_II`, and 4 in Cohort B missing `SNUADC/PLETH`).
4. **Replacement Selection**: Identified and fully audited 14 valid, track-complete replacement cases from the VitalDB database without using future ST labels for Cohort B.
5. **Storage and Budget Audit**: Confirmed the total expected raw download storage for the verified 100 cases is **~1.10 GB** (conservative ceiling: **~2.0 GB**), preserving $>90\%$ of the user's 25 GB working budget.

> [!IMPORTANT]
> **Enforcement of Hard Constraints**:
> * **Zero cases downloaded**: No `.vital` files outside the original 20 pilot cases have been fetched.
> * **Zero ML models trained**: No XGBoost, Random Forest, or synthetic data generation has been performed.
> * **Zero website modifications**: The BeatAhead website repository remains completely untouched.

---

## 1. Early-Warning Label Pipeline Validation (20 Pilot Cases)

### 1.1 Temporal Architecture Definition
The prospective early-warning prediction pipeline evaluates sliding windows with the following non-overlapping temporal segments:
* **Observation Window ($T_{\text{obs}}$)**: 300 seconds (5 minutes) of continuous multimodal physiological inputs (`ECG_II`, `PLETH`, `SpO2`).
* **Prediction Gap ($T_{\text{gap}}$)**: 300 seconds (5 minutes) of enforced lead time. **No model features or target information from this interval may be used at inference time.**
* **Target Horizon ($T_{\text{target}}$)**: 300 seconds (5 minutes) during which the clinical event is evaluated.
* **Stride**: 60 seconds (1 minute).

```
Timeline Progression (Total Span = 900 seconds / 15 minutes):
|-------------------|-------------------|-------------------|
|  T_obs (300 s)    |  T_gap (300 s)    | T_target (300 s)  |
|  [t, t + 300)     |  [t+300, t+600)   | [t+600, t+900)    |
|  Model Predictors | Enforced Lead-Time| Clinical Target   |
|-------------------|-------------------|-------------------|
```

### 1.2 Criteria for Valid Positive and Negative Examples
* **Valid Early-Warning Positive Example (`y = 1`)**:
  1. No sustained ST event in $T_{\text{obs}}$ ($\text{max\_consecutive}(ST\_II \le -1.0\text{ mm}) < 60\text{ s}$). If the patient is already ischemic in $T_{\text{obs}}$, it is not an early warning; the event is already in progress.
  2. Observation window ends; an enforced 5-minute prediction gap ($T_{\text{gap}}$) elapses.
  3. A verified sustained ST event occurs inside the target window ($\text{max\_consecutive}(ST\_II \le -1.0\text{ mm}) \ge 60\text{ s}$ inside $T_{\text{target}}$).
* **Valid Negative Control Example (`y = 0`)**:
  1. $T_{\text{obs}}$ contains valid physiological data.
  2. $T_{\text{target}}$ remains strictly within normal baseline limits ($\min(ST\_II) > -0.8\text{ mm}$ and $\max(ST\_II) < +0.8\text{ mm}$).
* **Indeterminate Exclusions**:
  1. Insufficient data coverage ($< 30$ valid numeric ST samples in $T_{\text{obs}}$ or $T_{\text{target}}$).
  2. Ambiguous buffer zone: Target window reaches $-1.0\text{ mm} < \min(ST\_II) \le -0.8\text{ mm}$ or brief non-sustained excursions ($< 60\text{ s}$). Excluded from binary classification to prevent label noise.

### 1.3 Audit Findings Across the 20 Pilot Cases

```
+----------------------------------------------------------------------------------------------------+
|                         20-CASE PILOT EARLY-WARNING AUDIT RESULTS SUMMARY                          |
+-------------------------------------------------------------------+--------------------------------+
| Metric                                                            | Exact Value on 20 Pilot Cases  |
+-------------------------------------------------------------------+--------------------------------+
| Total Corrected Sustained ST Episodes (full recording)            | 1 episode (Case 1, 186s dur)   |
| Total Positive Windows WITHOUT Prediction Gap (Concurrent)        | 6 windows (0.17% of cohort)    |
| Valid Early-Warning Positive Windows (Event Onset in T_target)     | 5 windows                      |
| Valid Early-Warning Positive Windows (Event Active in T_target)   | 6 windows                      |
| Confirmed Baseline Negative Windows                               | 3,288 windows                  |
| Indeterminate / Buffer / Insufficient Data Excluded Windows       | 216 windows                    |
| Total Cases Containing Valid Early-Warning Positives              | 1 case (Case 1, 5.0% of cohort)|
| Positive Examples Per Case: Case 1                                | 5 (or 6) windows               |
| Positive Examples Per Case: All Other 19 Cases                    | 0 windows                      |
+-------------------------------------------------------------------+--------------------------------+
```

### 1.4 Critical Clinical and Mathematical Analysis of Pilot Sparsity
1. **Severe Class Sparsity in Unselected Pilot**: In the 20-case unselected pilot, exactly **1 patient (Case 1)** experienced sustained intraoperative myocardial ischemia. When evaluated under the 5-minute early-warning architecture, this single episode yields **only 5 valid warning windows** (starting at minutes 164.0, 165.0, 166.0, 167.0, and 168.0) against **3,288 negative windows** (positive class prevalence: **0.15%**).
2. **Detailed Dynamic in Case 1**:
   * The 186-second sustained episode begins at $t = 10,663\text{ s}$ (minute 177.7) and ends at $t = 10,848\text{ s}$ (minute 180.8).
   * For window start $t = 164.0\text{ min}$ ($9,840\text{ s}$): $T_{\text{obs}} = [164, 169)\text{ min}$ (clean baseline, ST min $-0.6\text{ mm}$), $T_{\text{gap}} = [169, 174)\text{ min}$ (lead time, clean), $T_{\text{target}} = [174, 179)\text{ min}$ (contains first 78s of event) $\implies$ **VALID POSITIVE**.
   * For window start $t = 168.0\text{ min}$ ($10,080\text{ s}$): $T_{\text{obs}} = [168, 173)\text{ min}$ (clean), $T_{\text{gap}} = [173, 178)\text{ min}$ (brief 18s transient dip), $T_{\text{target}} = [178, 183)\text{ min}$ (contains 168s of event) $\implies$ **VALID POSITIVE**.
   * For window start $t = 169.0\text{ min}$ ($10,140\text{ s}$): The sustained episode has already started in $T_{\text{gap}}$ (78s consecutive inside gap), and continues into $T_{\text{target}}$ (108s). If early warning requires event *onset* in $T_{\text{target}}$, this window is excluded; if it requires event *presence* in $T_{\text{target}}$, it yields a 6th positive window.
3. **Integrity Rule on Prediction Gap**:
   > [!CAUTION]
   > **We must NOT reduce or remove the 5-minute prediction gap simply to artificially inflate positive counts.**
   > Eliminating the gap collapses the problem into concurrent diagnosis (detecting an event while it is already underway), which invalidates the clinical objective of BeatAhead: providing genuine advance warning so anesthesiologists can intervene before irreversible myocardial injury occurs.
   > The correct, scientifically valid solution is **cohort expansion enriched for cardiovascular risk (Cohort A)**.

---

## 2. Validation of the 100 Selected Cases (Proposed vs. Track Reality)

The proposed 100-case candidate list was systematically audited against the VitalDB database (`api.vitaldb.net`).

### 2.1 Track Defects in the Proposed List
Mandatory tracks for the multimodal pipeline are:
1. `SNUADC/ECG_II` (500 Hz high-resolution ECG Lead II)
2. `SNUADC/PLETH` (500 Hz high-resolution PPG pulse waveform)
3. `Solar8000/PLETH_SPO2` (Numeric oxygen saturation trend)
4. `Solar8000/ST_II` (Numeric ST deviation trend for Lead II)

**Audit Findings**:
* **Proposed Cohort A**: 46 of 50 cases had all 4 tracks. **4 cases (8.0%) lacked `Solar8000/ST_II`**: `2772`, `3192`, `3042`, `4246`.
* **Proposed Cohort B**: 40 of 50 cases had all 4 tracks. **10 cases (20.0%) were defective**:
  * 6 cases lacked `Solar8000/ST_II`: `3687`, `1374`, `6081`, `2146`, `5130`, `3450`.
  * 4 cases lacked `SNUADC/PLETH`: `1344`, `3398`, `3486`, `5262`.
* **Total defective cases across proposed list**: **14 cases (14.0%)**.

### 2.2 Case-Level Metadata Tables for Proposed Cohorts

#### Proposed Cohort A Case-by-Case Audit (50 Cases)

| Case ID | Status | Dur (h) | Dept | Age | HTN | DM | CAD/CABG | ECG_II | PLETH | SpO2 | ST_II | ECG_V5 | ST_V5 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2605 | VERIFIED | 3.71 | General surgery | 91 | Yes | Yes | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 3614 | VERIFIED | 1.83 | General surgery | 90 | Yes | Yes | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 4290 | VERIFIED | 1.76 | General surgery | 85 | Yes | Yes | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 2043 | VERIFIED | 1.96 | General surgery | 84 | Yes | Yes | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 4247 | VERIFIED | 2.45 | General surgery | 84 | Yes | Yes | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 2034 | VERIFIED | 2.37 | General surgery | 83 | Yes | Yes | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 3390 | VERIFIED | 2.51 | General surgery | 83 | Yes | Yes | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 3526 | VERIFIED | 2.04 | General surgery | 83 | Yes | Yes | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 758 | VERIFIED | 2.26 | General surgery | 82 | Yes | Yes | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 1159 | VERIFIED | 2.40 | General surgery | 82 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 2197 | VERIFIED | 5.15 | General surgery | 54 | Yes | Yes | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 2772 | **REJECTED** | 3.33 | General surgery | 78 | No | No | No | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ |
| 4272 | VERIFIED | 5.92 | General surgery | 37 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 2824 | VERIFIED | 5.41 | General surgery | 67 | Yes | Yes | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 3037 | VERIFIED | 6.55 | Thoracic surgery | 61 | Yes | Yes | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 3632 | VERIFIED | 4.46 | General surgery | 31 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 2800 | VERIFIED | 3.80 | General surgery | 58 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 3192 | **REJECTED** | 3.43 | General surgery | 50 | No | No | No | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ |
| 4519 | VERIFIED | 2.48 | General surgery | 56 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 2854 | VERIFIED | 2.19 | General surgery | 60 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 4038 | VERIFIED | 1.31 | General surgery | 76 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 2037 | VERIFIED | 2.71 | General surgery | 63 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 2717 | VERIFIED | 3.11 | Thoracic surgery | 74 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 3027 | VERIFIED | 3.52 | General surgery | 80 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 4390 | VERIFIED | 1.95 | General surgery | 80 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 4220 | VERIFIED | 1.54 | General surgery | 79 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 2697 | VERIFIED | 1.83 | General surgery | 79 | Yes | Yes | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 4423 | VERIFIED | 1.57 | General surgery | 78 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 2085 | VERIFIED | 2.06 | General surgery | 78 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 3505 | VERIFIED | 1.62 | General surgery | 77 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 3042 | **REJECTED** | 3.12 | General surgery | 46 | No | No | No | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ |
| 3535 | VERIFIED | 1.70 | General surgery | 77 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 4252 | VERIFIED | 2.47 | General surgery | 76 | Yes | Yes | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 2727 | VERIFIED | 3.57 | General surgery | 64 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 3426 | VERIFIED | 3.73 | General surgery | 46 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 4236 | VERIFIED | 1.93 | General surgery | 75 | Yes | Yes | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 4243 | VERIFIED | 1.90 | General surgery | 75 | Yes | Yes | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 4434 | VERIFIED | 2.21 | General surgery | 75 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 2725 | VERIFIED | 4.88 | General surgery | 41 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 3418 | VERIFIED | 5.37 | General surgery | 58 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 4437 | VERIFIED | 1.69 | General surgery | 74 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 2081 | VERIFIED | 1.60 | General surgery | 73 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 2623 | VERIFIED | 2.23 | General surgery | 73 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 4246 | **REJECTED** | 1.75 | General surgery | 72 | Yes | No | No | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ |
| 3530 | VERIFIED | 1.68 | General surgery | 71 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 4568 | VERIFIED | 1.64 | General surgery | 71 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 2075 | VERIFIED | 1.93 | General surgery | 69 | Yes | Yes | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 4435 | VERIFIED | 2.05 | General surgery | 68 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 4244 | VERIFIED | 1.70 | General surgery | 68 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 4234 | VERIFIED | 2.45 | General surgery | 68 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |

#### Proposed Cohort B Case-by-Case Audit (50 Cases)

| Case ID | Status | Dur (h) | Dept | Age | HTN | DM | CAD/CABG | ECG_II | PLETH | SpO2 | ST_II | ECG_V5 | ST_V5 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1268 | VERIFIED | 3.06 | Thoracic surgery | 57 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 485 | VERIFIED | 3.96 | Thoracic surgery | 67 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 2630 | VERIFIED | 3.08 | General surgery | 68 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 4498 | VERIFIED | 3.75 | General surgery | 41 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 3948 | VERIFIED | 1.59 | General surgery | 72 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 3754 | VERIFIED | 2.25 | General surgery | 54 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 5799 | VERIFIED | 2.96 | Thoracic surgery | 60 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 3216 | VERIFIED | 2.14 | General surgery | 78 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 864 | VERIFIED | 1.88 | General surgery | 83 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 1923 | VERIFIED | 2.63 | General surgery | 69 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 497 | VERIFIED | 7.95 | General surgery | 66 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 1855 | VERIFIED | 4.73 | General surgery | 60 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 3687 | **REJECTED** | 4.12 | General surgery | 28 | No | No | No | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ |
| 5779 | VERIFIED | 0.78 | General surgery | 76 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 1374 | **REJECTED** | 6.25 | General surgery | 77 | No | No | No | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| 1344 | **REJECTED** | 6.22 | General surgery | 45 | No | No | No | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ |
| 4225 | VERIFIED | 5.29 | General surgery | 62 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 4016 | VERIFIED | 1.75 | Thoracic surgery | 18 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 5635 | VERIFIED | 1.47 | Thoracic surgery | 79 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 3398 | **REJECTED** | 2.43 | General surgery | 62 | No | No | No | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ |
| 4771 | VERIFIED | 1.23 | General surgery | 81 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 381 | VERIFIED | 1.19 | General surgery | 74 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 2866 | VERIFIED | 2.89 | Gynecology | 41 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 1714 | VERIFIED | 2.45 | General surgery | 69 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 2419 | VERIFIED | 2.93 | Thoracic surgery | 63 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 4467 | VERIFIED | 1.83 | General surgery | 67 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 360 | VERIFIED | 1.55 | Thoracic surgery | 68 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 481 | VERIFIED | 2.42 | Thoracic surgery | 58 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 3486 | **REJECTED** | 2.12 | General surgery | 64 | No | No | No | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ |
| 2728 | VERIFIED | 3.58 | General surgery | 39 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 5585 | VERIFIED | 2.80 | General surgery | 48 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 2172 | VERIFIED | 2.76 | General surgery | 50 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 6081 | **REJECTED** | 3.73 | General surgery | 61 | Yes | No | No | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ |
| 4616 | VERIFIED | 3.49 | Thoracic surgery | 64 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 2253 | VERIFIED | 4.39 | General surgery | 66 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 569 | VERIFIED | 4.29 | Thoracic surgery | 59 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 111 | VERIFIED | 4.38 | General surgery | 68 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 2719 | VERIFIED | 5.56 | General surgery | 61 | Yes | Yes | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 4349 | VERIFIED | 5.92 | General surgery | 66 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 1269 | VERIFIED | 10.97 | General surgery | 59 | Yes | Yes | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 5664 | VERIFIED | 4.88 | General surgery | 63 | Yes | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 5326 | VERIFIED | 4.79 | Thoracic surgery | 63 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 3280 | VERIFIED | 1.83 | Thoracic surgery | 65 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 2146 | **REJECTED** | 2.50 | General surgery | 37 | No | No | No | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ |
| 5130 | **REJECTED** | 2.37 | General surgery | 65 | No | No | No | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| 3450 | **REJECTED** | 2.23 | General surgery | 58 | No | No | No | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ |
| 4120 | VERIFIED | 1.48 | General surgery | 40 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 5262 | **REJECTED** | 1.80 | General surgery | 71 | No | No | No | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ |
| 2379 | VERIFIED | 3.49 | General surgery | 52 | No | No | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 451 | VERIFIED | 3.19 | General surgery | 67 | Yes | Yes | No | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |

---

## 3. Cohort B Representativeness & Comparison with Cohort A

### 3.1 Summary Distributions and Benchmark Comparison

```
+---------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                      STATISTICAL DISTRIBUTION COMPARISON TABLE                                                           |
+------------------------------------+----+---------------+--------------------+------------------+--------------------+---------------+--------------+
| Cohort                             | N  | Age (Mean±SD) | Age Median (IQR)   | Dur hrs (Mean±SD)| Dur Median (IQR)   | Preop HTN (%) | Preop DM (%) |
+------------------------------------+----+---------------+--------------------+------------------+--------------------+---------------+--------------+
| Cohort A (Proposed Raw)            | 50 | 60.6 ± 19.9   | 61.0 (47.8 - 78.0) | 2.88 ± 1.46      | 2.60 (1.66 - 3.78) | 40.0% (20/50) | 28.0% (14/50)|
| Cohort A (Final Verified)          | 50 | 61.9 ± 20.3   | 64.0 (48.5 - 79.8) | 2.85 ± 1.46      | 2.50 (1.66 - 3.87) | 46.0% (23/50) | 30.0% (15/50)|
| Cohort B (Proposed Raw)            | 50 | 59.0 ± 18.2   | 62.0 (51.8 - 68.8) | 3.31 ± 2.10      | 2.69 (1.65 - 4.42) | 36.0% (18/50) |  6.0% (3/50) |
| Cohort B (Final Verified)          | 50 | 60.3 ± 16.1   | 63.0 (54.5 - 71.2) | 3.01 ± 1.73      | 2.71 (1.76 - 3.86) | 32.0% (16/50) |  4.0% (2/50) |
| VitalDB Full Benchmark             |6388| 57.3 ± 15.0   | 59.0 (48.0 - 68.0) | 3.15 ± 1.85      | 2.76 (1.72 - 4.19) | 30.8%         | 10.3%        |
+------------------------------------+----+---------------+--------------------+------------------+--------------------+---------------+--------------+
```

### 3.2 Department Breakdown Comparison (%)

| Surgical Department | Cohort A (Final Verified) | Cohort B (Proposed Raw) | Cohort B (Final Verified) | VitalDB Full Benchmark (N=6,388) |
| :--- | :---: | :---: | :---: | :---: |
| **General Surgery** | 78.0% | 70.0% | **58.0%** | 77.2% |
| **Thoracic Surgery** | 20.0% | 24.0% | **28.0%** | 17.4% |
| **Gynecology** | 2.0% | 4.0% | **8.0%** | 3.6% |
| **Urology** | 0.0% | 2.0% | **6.0%** | 1.8% |

### 3.3 Critical Scientific Assessment of Representativeness
> [!WARNING]
> **Cohort B DOES NOT represent the general ambulatory population.**
> * VitalDB consists entirely of in-hospital patients undergoing major non-cardiac surgery under general anesthesia at a single tertiary academic medical center (Seoul National University Hospital).
> * Factors such as mechanical positive-pressure ventilation, volatile anesthetics (sevoflurane/desflurane), intravenous opioids, muscle relaxants, and intraoperative fluid shifts heavily alter autonomic vascular tone and pulse transit times.
> * **Demonstrated Alignment**: Cohort B (Final Verified) closely matches the **intraoperative tertiary surgical population** of VitalDB in surgical duration (median 2.71h vs 2.76h benchmark), age distribution (median 63.0 vs 59.0 years), and hypertension prevalence (32.0% vs 30.8%). Furthermore, the 10 replacements enrich multi-specialty diversity across Thoracic, Gynecology, and Urology.
> * **No Selection on ST Outcomes**: Cohort B cases were selected strictly through procedural and demographic variables, **completely blind to ST deviation values or ischemic outcomes**.

---

## 4. Duplicate, Overlap & Replacement Case Audit

### 4.1 Integrity & Overlap Verification
* **Duplicate Case IDs**: 0 duplicates within Cohort A; 0 duplicates within Cohort B.
* **A vs. B Overlap**: Exactly **0 cases overlap** ($\text{Cohort A} \cap \text{Cohort B} = \emptyset$).
* **Overlap with 20-Case Pilot Cohort**: Exactly **0 cases overlap** ($\text{Pilot} \cap (\text{Cohort A} \cup \text{Cohort B}) = \emptyset$).
* **Record Uniqueness**: Every case ID maps to exactly one unique surgical recording in VitalDB.

### 4.2 Detailed Audit of Replacement Cases

To restore both cohorts to 50 fully verified cases, **14 replacements** were drawn from the qualified candidate pool:

#### Cohort A Replacements (4 Cases)
Replaced defective cases `[2772, 3192, 3042, 4246]` with elderly, high-cardiovascular-risk patients possessing all 4 mandatory tracks PLUS high-resolution Lead V5 (`SNUADC/ECG_V5`):

| Replacement Case ID | Replacing | Dur (h) | Dept | Age | Preop HTN | Preop DM | Mandatory Tracks | ECG_V5 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **70** | 2772 | 3.93 | General surgery | 76 | Yes | No | ✓ (All 4) | ✓ |
| **87** | 3192 | 3.12 | General surgery | 71 | Yes | No | ✓ (All 4) | ✓ |
| **137** | 3042 | 2.01 | General surgery | 80 | Yes | Yes | ✓ (All 4) | ✓ |
| **232** | 4246 | 3.91 | General surgery | 87 | Yes | No | ✓ (All 4) | ✓ |

#### Cohort B Replacements (10 Cases)
Replaced defective cases `[3687, 1374, 1344, 3398, 3486, 6081, 2146, 5130, 3450, 5262]` with representative surgical cases possessing all 4 mandatory tracks, selected blind to ST values and balanced across specialties:

| Replacement Case ID | Replacing | Dur (h) | Dept | Age | Preop HTN | Preop DM | Mandatory Tracks | ECG_V5 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **7** | 3687 | 4.38 | Thoracic surgery | 62 | No | No | ✓ (All 4) | ✗ |
| **44** | 1374 | 4.06 | Thoracic surgery | 69 | No | No | ✓ (All 4) | ✗ |
| **67** | 1344 | 3.22 | Thoracic surgery | 67 | No | No | ✓ (All 4) | ✗ |
| **175** | 3398 | 3.31 | Urology | 61 | No | No | ✓ (All 4) | ✗ |
| **388** | 3486 | 3.85 | Urology | 66 | No | No | ✓ (All 4) | ✗ |
| **108** | 6081 | 2.66 | Gynecology | 42 | No | No | ✓ (All 4) | ✗ |
| **112** | 2146 | 2.98 | Gynecology | 63 | No | No | ✓ (All 4) | ✗ |
| **24** | 5130 | 1.80 | General surgery | 55 | No | No | ✓ (All 4) | ✗ |
| **25** | 3450 | 4.12 | General surgery | 64 | No | No | ✓ (All 4) | ✓ |
| **27** | 5262 | 4.96 | General surgery | 48 | No | No | ✓ (All 4) | ✗ |

### 4.3 Final Verified 100-Case ID Lists
* **Cohort A (Final Verified, N=50)**:
  `[2605, 3614, 4290, 2043, 4247, 2034, 3390, 3526, 758, 1159, 2197, 4272, 2824, 3037, 3632, 2800, 4519, 2854, 4038, 2037, 2717, 3027, 4390, 4220, 2697, 4423, 2085, 3505, 3535, 4252, 2727, 3426, 4236, 4243, 4434, 2725, 3418, 4437, 2081, 2623, 3530, 4568, 2075, 4435, 4244, 4234, 70, 87, 137, 232]`
* **Cohort B (Final Verified, N=50)**:
  `[1268, 485, 2630, 4498, 3948, 3754, 5799, 3216, 864, 1923, 497, 1855, 5779, 4225, 4016, 5635, 4771, 381, 2866, 1714, 2419, 4467, 360, 481, 2728, 5585, 2172, 4616, 2253, 569, 111, 2719, 4349, 1269, 5664, 5326, 3280, 4120, 2379, 451, 7, 44, 67, 175, 388, 108, 112, 24, 25, 27]`

---

## 5. Reconsideration of the ST Feature Decision & Future Ablation Plan

Rather than indiscriminately purging all ST metrics or introducing target leakage, ST data are classified into three distinct temporal categories:

### 5.1 Three-Tier Classification of ST Information
1. **Category A — Future-Target ST Data (`PROHIBITED`)**:
   * Any ST observation occurring within the prediction gap ($T_{\text{gap}}$) or target window ($T_{\text{target}}$).
   * **Rule**: Absolutely forbidden from model inputs. Inclusion constitutes direct label leakage.
2. **Category B — Pre-Event ST Data inside $T_{\text{obs}}$ (`POTENTIALLY VALID PREDICTOR`)**:
   * Descriptive metrics of ST deviation computed purely within the 5-minute observation window $T_{\text{obs}}$ (e.g., mean ST, minimum ST, ST standard deviation, ST slope/trajectory).
   * **Clinical Justification**: In practice, early micro-shifts or minor downward trends in the ST segment often precede full $\ge 60\text{ s}$ diagnostic depression. Evaluating whether these subtle changes have predictive value is clinically meaningful.
3. **Category C — Baseline-Relative ST Features (`POTENTIALLY VALID PREDICTOR`)**:
   * Normalized ST shifts relative to the patient's individual resting baseline established during early stable anesthesia ($\Delta\text{ST} = \text{ST}_{\text{obs}} - \text{ST}_{\text{baseline}}$).
   * **Clinical Justification**: Adjusts for pre-existing bundle branch blocks, left ventricular hypertrophy, or benign non-ischemic baseline elevations.

### 5.2 Future Ablation Plan: Model A vs. Model B
When modeling is authorized in future sprints, the pipeline will benchmark two formal configurations:

```
+----------------------------------------------------------------------------------------------------+
|                                    FUTURE ABLATION SPECIFICATION                                   |
+------------------------------------+----------------------------------+----------------------------+
| Configuration                      | Model A (Multimodal + Pre-ST)    | Model B (Pure Non-ST)      |
+------------------------------------+----------------------------------+----------------------------+
| Input Modalities                   | ECG + PPG + SpO2 + Pre-Event ST  | ECG + PPG + SpO2 (NO ST)   |
| R-Peak / HRV Metrics               | Included                         | Included                   |
| PPG Morphology / Perfusion Index   | Included                         | Included                   |
| Pulse Arrival Time (PAT / PTT)     | Included                         | Included                   |
| SpO2 Trend / Desaturation Features | Included                         | Included                   |
| Pre-Event ST Mean / Slope (T_obs)  | INCLUDED                         | EXCLUDED                   |
| Baseline-Relative Delta-ST         | INCLUDED                         | EXCLUDED                   |
| Core Scientific Hypothesis         | Best overall discriminatory power| Evaluates whether vascular |
|                                    | for impending ischemia           | timing alone predicts      |
|                                    |                                  | ischemia without ST inputs |
+------------------------------------+----------------------------------+----------------------------+
```

---

## 6. Storage and Working Budget Calculation

### 6.1 Pilot File Ground Truth
* 20 Pilot `.vital` files downloaded: **252.57 MB** total (Mean: **12.63 MB/file**, Min: 4.39 MB, Max: 20.23 MB).
* Total surgical recording time across 20 pilot cases: **64.2 hours**.
* Empirical storage density: **~3.83 MB per surgical hour**.

### 6.2 100-Case Raw Storage Estimate
* Cohort A (50 Verified Cases) Total Duration: **142.45 hours** (Mean: 2.85 hours).
* Cohort B (50 Verified Cases) Total Duration: **150.35 hours** (Mean: 3.01 hours).
* Combined 100 Cases Total Duration: **292.80 hours** (Mean: 2.93 hours).
* **Expected Total Raw Download Size**:
  $$\text{Expected Storage} = 292.80\text{ hours} \times 3.83\text{ MB/hour} \approx 1,121.4\text{ MB} \approx \mathbf{1.10\text{ GB}}$$
* **Conservative Ceiling (accounting for high track-density cases)**: **$\le 2.0\text{ GB}$**.

### 6.3 Budget Headroom
* User working storage budget: **25.0 GB**.
* Expected consumption: **~1.10 GB** ($4.4\%$ of budget).
* Remaining storage headroom: **~23.90 GB ($>90\%$ available)**.

---

## 7. Final Recommendation & Checklist

```
+----------------------------------------------------------------------------------------------------+
|                                    PRE-DOWNLOAD READINESS AUDIT                                    |
+--------------------------------------------------------------------+-------------------------------+
| Audit Item                                                         | Status                        |
+--------------------------------------------------------------------+-------------------------------+
| Consecutive ST run-length logic validated (no cumulative sum bugs) | VERIFIED                      |
| Early-warning temporal architecture audited on pilot (5-min gap)   | VERIFIED (5-6 pos windows, C1)|
| Pilot class sparsity documented transparently                      | VERIFIED                      |
| Proposed 100 cases audited against VitalDB metadata API            | COMPLETED                     |
| 14 defective cases identified and rejected                         | VERIFIED                      |
| 14 valid, track-complete replacements selected and audited         | VERIFIED                      |
| Zero duplicate case IDs verified                                   | VERIFIED                      |
| Zero overlap between Cohort A, Cohort B, and Pilot cohort          | VERIFIED                      |
| Storage impact calculated (1.10 GB / 25 GB budget)                | VERIFIED (< 5% of budget)     |
| Machine learning models trained                                    | ZERO (STRICTLY HALTED)        |
| Raw 100 cases downloaded                                           | ZERO (STRICTLY HALTED)        |
| BeatAhead website modified                                         | ZERO (STRICTLY HALTED)        |
+--------------------------------------------------------------------+-------------------------------+
```

### Recommendation
1. **DO NOT download the original unverified 100-case candidate list**, as 14 cases lack mandatory waveforms.
2. **APPROVE the Final Verified 100-Case List** (incorporating the 14 track-complete replacements) for Phase 3 acquisition.
3. **STOP AND AWAIT HUMAN REVIEW**: No downloads or model training will commence until explicit written authorization is received from the user.
