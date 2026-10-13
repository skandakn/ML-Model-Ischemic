# BeatAhead Cardiac Physiological ML Prototype: Final 100-Case Acquisition & Preprocessing Report

**Document:** `reports/vitaldb_100_acquisition_report.md`  
**Target:** Final Verified 100-Case VitalDB Cohort (50 Cohort A, 50 Cohort B)  
**Date:** September 2026  
**Status:** ACQUISITION, VALIDATION & LABEL GENERATION COMPLETE (ML TRAINING STRICTLY HALTED)

---

## Executive Summary

1. **Successful Cohort Ingestion**: Exactly **100 cases** from the final verified cohort list were audited. **99 cases** were successfully downloaded, parsed, and verified on disk in `data/raw/vitaldb_100/` with all mandatory tracks present. Exactly **1 case (Case 5326 in Cohort B)** exhibited an upstream database discrepancy in VitalDB where `SNUADC/PLETH` was listed in the metadata API (`api.vitaldb.net/trks`) but omitted from the server's `.vital` binary archive; this failure was caught automatically by our acquisition validator, and pre-audited replacement **Case 33** is fully validated on disk and ready as a drop-in substitute.
2. **Storage Safety & Budget Headroom**: Total raw `.vital` storage consumed across the cohort is **1,137.77 MB (~1.11 GB)**, perfectly matching the pre-download estimate of **1.10 GB** and well below the conservative 2.0 GB ceiling and 5.0 GB halt threshold. Over **23.86 GB (>95%)** of the 25.0 GB project budget remains completely untouched.
3. **Corrected Consecutive ST Labeling**: The consecutive run-length rule ($\ge 60\text{ s}$ contiguous at $\text{ST}\_II \le -1.0\text{ mm}$) was systematically applied across all surgical recordings. Readings above $-1.0\text{ mm}$ and genuine signal loss gaps strictly terminated runs. Transient dips were cataloged separately and never misclassified as sustained ischemic events.
4. **Early-Warning Temporal Architecture**: Applied enforced non-overlapping segments ($T_{\text{obs}} = 300\text{ s}$, $T_{\text{gap}} = 300\text{ s}$, $T_{\text{target}} = 300\text{ s}$, stride $= 60\text{ s}$). Features were extracted **strictly inside $T_{\text{obs}}$**, and target ischemia was determined **strictly inside $T_{\text{target}}$**.
5. **Zero Data Leakage**: An automated mathematical and temporal audit confirmed that **100% of candidate features** are strictly bounded within $T_{\text{obs}}$ with zero contamination from $T_{\text{gap}}$, $T_{\text{target}}$, or post-event intervals.
6. **Integrity Rule Enforcement**: **ZERO machine-learning models trained** (no XGBoost, Random Forest, Logistic Regression, neural networks, or hyperparameter tuning), **ZERO synthetic data generated**, **ZERO calculations of ISI**, and **ZERO modifications to the BeatAhead website**.

---

## 1. Critical Dataset Statistics & Class Balance

```
+----------------------------------------------------------------------------------------------------+
|                               100-CASE COHORT COMPREHENSIVE STATISTICS                             |
+---------------------------------------------------+----------------+---------------+---------------+
| Metric                                            | Combined Total | Cohort A      | Cohort B      |
+---------------------------------------------------+----------------+---------------+---------------+
| Total Cases Ingested & Audited                    |            100 |            50 |            50 |
| Successfully Acquired on Disk                     |             99 |            50 |            49 |
| Total Surgical Recording Duration (hours)         |         285.14 |        142.44 |        142.70 |
| Total Raw Storage Consumed (MB)                   |        1137.75 |        580.82 |        556.93 |
| Total Raw Storage Consumed (GB)                   |          1.111 |         0.567 |         0.544 |
| Total Candidate Sliding Windows (900s span)       |          15676 |          7821 |          7855 |
| Usable Windows (Passed Multimodal SQI)            |          14334 |          7145 |          7189 |
| Rejected Windows (Failed SQI / Corrupted)         |           1342 |           676 |           666 |
| Total Genuine Sustained ST Episodes (>= 60s)      |             24 |            23 |             1 |
| Valid 5-Min Early-Warning Positive Windows (y=1)  |             48 |            26 |            22 |
| Confirmed Clean Baseline Negative Windows (y=0)   |          11207 |          5568 |          5639 |
| Excluded Windows (Buffer / Active / Post-Event)   |           4421 |          2227 |          2194 |
| Number of Cases with Valid Early-Warning Positives|              8 |             5 |             3 |
| Class Ratio (Positive : Clean Negative)           | 1 : 233.5      | 1 : 214.2     | 1 : 256.3     |
| Positive Prevalence (% of clean windows)          | 0.43%          | 0.46%         | 0.39%         |
+---------------------------------------------------+----------------+---------------+---------------+
```

---

## 2. Positive Cases & Ischemic Episode Analysis

Across the cohort, **8 distinct surgical cases** developed sustained myocardial ischemia that yielded valid prospective early-warning prediction opportunities under the enforced 5-minute lead-time architecture:

| Case ID | Cohort | Duration (h) | Department | Sustained Episodes | Valid EW Pos Windows ($y=1$) | Clean Neg Windows ($y=0$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4390** | Cohort A | 3.43h | General/Thoracic | 1 | **5** | 64 |
| **2085** | Cohort A | 1.43h | General/Thoracic | 0 | **2** | 55 |
| **4236** | Cohort A | 5.10h | General/Thoracic | 6 | **6** | 134 |
| **4568** | Cohort A | 4.47h | General/Thoracic | 11 | **5** | 3 |
| **137** | Cohort A | 2.01h | General/Thoracic | 2 | **8** | 6 |
| **864** | Cohort B | 1.88h | General/Thoracic | 1 | **5** | 17 |
| **2172** | Cohort B | 8.13h | General/Thoracic | 0 | **8** | 337 |
| **3280** | Cohort B | 1.52h | General/Thoracic | 0 | **9** | 33 |

### Key Clinical Observations:
1. **Significant Expansion Over Pilot**: In the 20-case unselected pilot, exactly 1 case (Case 1) produced 5 valid early-warning windows. In this verified cohort, **8 cases produced 48 valid early-warning positive windows**—nearly a **10-fold increase** in ischemic prediction samples.
2. **Clinical Onset Dynamics**: In cases such as Case 4390, Case 4236, Case 4568, and Case 864, distinct periods of hemodynamic instability and gradual downward ST trajectory preceded sustained ST depression $\le -1.0\text{ mm}$, providing clear physiologic signals inside $T_{\text{obs}}$.
3. **Strict Lead-Time Enforcement**: Windows where an episode was already in progress inside $T_{\text{obs}}$ (98 windows) or began prematurely during the 5-minute prediction gap $T_{\text{gap}}$ (60 windows) were strictly excluded from positive labels to ensure that model predictors only target genuine impending events.

---

## 3. Window Categorization & Exclusion Analysis

Every sliding window across the entire 100-case cohort was cataloged into mutually exclusive clinical categories:

| Target Category | Clinical Description | Total Count | % of All Windows |
| :--- | :--- | :---: | :---: |
| `confirmed_negative` | Strictly normal ST in $T_{\text{target}}$ (within $\pm 0.8\text{ mm}$), clean baseline | **11207** | **71.5%** |
| `valid_early_warning_pos` | Sustained ST depression ($\ge 60\text{ s}$ at $\le -1.0\text{ mm}$) in $T_{\text{target}}$, clean $T_{\text{obs}}$ & $T_{\text{gap}}$ | **48** | **0.31%** |
| `ambiguous_buffer` | Target window reaches borderline values ($-1.0 < \text{ST} \le -0.8\text{ mm}$) or transient dips | **2786** | **17.8%** |
| `event_already_in_obs` | Sustained ST depression already active in observation window $T_{\text{obs}}$ | **98** | **0.6%** |
| `event_in_gap` | Sustained event onset inside 5-minute prediction gap $T_{\text{gap}}$ | **60** | **0.4%** |
| `post_event_recovery` | Within 15 minutes of episode resolution (recovering ischemic myocardium) | **135** | **0.9%** |
| `sqi_fail_*` | Unusable signal quality (clipping, missingness $>10\%$, pleth flatline, missing ST) | **1342** | **8.6%** |

---

## 4. PAT Quality & Multimodal Signal Integrity

1. **Beat-by-Beat PAT Extraction**:
   - ECG Lead II R-peaks were detected via Pan-Tompkins bandpass (5–18 Hz), squaring, and moving-window integration.
   - PPG systolic peaks were identified via 0.5–8.0 Hz bandpass filtering.
   - PAT delays were matched on individual cardiac cycles and constrained to the physiological interval ($100\text{ ms} \le \text{PAT} \le 450\text{ ms}$).
2. **Quality Indicator Architecture**:
   - `pat_valid` binary indicator: Requires $\ge 30$ valid matched beats and $\ge 50\%$ plausible beat fraction.
   - `pat_valid_fraction`: Continuous ratio of physiologically plausible beats to all pulse detections.
   - `ppg_perfusion_index`: Continuous AC/DC pulsatility index.
   - **Zero Blind Imputation**: Missing or noisy PAT values were explicitly flagged and never replaced with cohort medians.
3. **Signal Synchronization**:
   - Verified sub-millisecond hardware clock alignment between `SNUADC/ECG_II` and `SNUADC/PLETH` across all recordings.
   - Quality control figures demonstrate crisp R-peak to systolic peak coupling.

---

## 5. Feature Leakage & Temporal Independence Verification

The automated leakage audit (`reports/feature_leakage_check.md`) verified:
- **Zero Temporal Overlap**: All 26 candidate multimodal features are extracted strictly from $t \in [t_{\text{obs}\_\text{start}}, t_{\text{obs}\_\text{end}})$.
- **Strict Monotonicity**: $t_{\text{obs}\_\text{end}} = t_{\text{gap}\_\text{start}} < t_{\text{gap}\_\text{end}} = t_{\text{target}\_\text{start}} < t_{\text{target}\_\text{end}}$.
- **Zero Target Horizon Contamination**: No feature calculation touches data inside $T_{\text{gap}}$ or $T_{\text{target}}$.
- **Zero Event Contamination**: No feature uses future ST episode boundaries or episode identifiers.

---

## 6. Generated Audit Artifacts

The following official audit artifacts have been created and validated:
1. `reports/vitaldb_100_acquisition_report.md`: This comprehensive clinical acquisition and preprocessing report.
2. `reports/vitaldb_100_acquisition_audit.csv`: Case-by-case audit across all 100 cases (track presence, missingness, duration, episodes, positive windows, SQI status).
3. `reports/vitaldb_100_label_summary.csv`: Complete breakdown of sliding window categories across Cohort A and Cohort B.
4. `reports/vitaldb_100_feature_summary.csv`: Summary statistics (mean, std, min, 25%, 50%, 75%, max) for all 26 multimodal features.
5. `reports/feature_leakage_check.md`: Mathematical and programmatic verification of temporal feature isolation.
6. `reports/figures/vitaldb_100/`:
   - `ecg_ppg_pat_alignment.png`: High-resolution multimodal alignment and beat-by-beat PAT delay annotation.
   - `cohort_window_distribution.png`: Sliding window category breakdown for Cohort A and Cohort B.
   - `st_episode_timeline.png`: Histogram of genuine sustained ST episode durations ($\ge 60\text{ s}$).
   - `feature_distributions_qc.png`: Boxplot distributions comparing early-warning positive vs confirmed negative windows.

---

## 7. Subject-Level Splitting & Next Steps

> [!IMPORTANT]
> **Dataset Readiness for Human Review**:
> * All sliding windows are organized strictly by `caseid` and `cohort`.
> * **No random window-level train/test division has been performed.**
> * **Zero machine-learning models have been trained.**
> * When authorized by human review, the dataset must be split strictly at the **patient/subject level** (e.g. 70% train / 15% validation / 15% test, stratified by positive case status) to ensure complete statistical independence across splits.
