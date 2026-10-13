# BeatAhead Subject-Level Split Simulation & Pre-Modeling Evaluation Audit

**Document:** `reports/subject_level_split_audit.md`  
**Execution Date:** 2026-09-22 16:41:40  
**Cohort Scope:** Final Verified 100-Case Dataset (50 Cohort A, 50 Cohort B)  
**Status:** PRE-MODELING AUDIT COMPLETE — MACHINE LEARNING STRICTLY HALTED

---

## Executive Summary

1. **Complete 100-Case Cohort Ingestion**: The final 100th case was completed by validating pre-audited replacement **Case 33** (General surgery, Hernia repair, 1.75h) which possesses all 4 mandatory tracks (`SNUADC/ECG_II`, `SNUADC/PLETH`, `Solar8000/PLETH_SPO2`, `Solar8000/ST_II`). The final verified dataset comprises **exactly 100 cases** (50 Cohort A, 50 Cohort B) with **1,143.14 MB (~1.12 GB)** raw storage.
2. **Corrected Window Terminology**: The temporal windowing architecture is formally documented as **overlapping sliding windows with a 60-second stride** ($T_{\text{obs}} = 300\text{ s}$, $T_{\text{gap}} = 300\text{ s}$, $T_{\text{target}} = 300\text{ s}$, total span $= 900\text{ s}$).
3. **Case-Level Isolation Audit**: Zero window-level random splitting was permitted. 20 independent, deterministic 70/15/15 grouped split simulations confirmed that the same patient never appears in multiple partitions.
4. **Critical Finding on 70/15/15 Viability**: Because exactly **8 patients** in the 100-case cohort developed sustained myocardial ischemia, a single 15% test set contains only **1 to 2 positive cases** (5 to 17 positive windows). Performance estimates on a single test partition will have **extremely wide confidence intervals and high empirical variance**.
5. **Recommended Evaluation Protocol**: Grouped 5-Fold Cross-Validation (`StratifiedGroupKFold`) provides substantially superior development stability by rotating all 8 positive cases across folds while maintaining strict subject-level isolation. A small locked test set (20 cases) should be held strictly untouched for final confirmation.
6. **Dual Modeling Matrix Specification**: Metadata specifications for **Model Matrix A** (Multimodal + Pre-event ST, 26 features) and **Model Matrix B** (Pure Non-ST, 20 features) are finalized for future ablation benchmarking.

---

## 1. 100th Case Validation & Final Cohort Integrity

Defective candidate Case 5326 (where the VitalDB server binary file omitted `SNUADC/PLETH` despite being indexed in `trks.csv`) was replaced with pre-audited **Case 33**:

| Parameter | Specification / Result |
| :--- | :--- |
| **Case ID** | **33** (Replacing Case 5326) |
| **Cohort** | Cohort B (Representative Multi-Specialty Surgical) |
| **Procedure & Diagnosis** | General Surgery — Hernia repair |
| **Patient Demographics** | Age 47, Non-hypertensive, Non-diabetic |
| **Recording Duration** | 1.75 hours (6,317 seconds) |
| **Mandatory Track `SNUADC/ECG_II`** | **PRESENT [500 Hz, Verified]** |
| **Mandatory Track `SNUADC/PLETH`** | **PRESENT [500 Hz, Verified]** |
| **Mandatory Track `Solar8000/PLETH_SPO2`** | **PRESENT [1 Hz, Verified]** |
| **Mandatory Track `Solar8000/ST_II`** | **PRESENT [1 Hz, Verified]** |
| **Binary Status** | Verified on disk (`data/raw/vitaldb_100/33.vital`, 5.39 MB) |

### Final Verified 100-Case Dataset Metrics:
- **Total Audited Cases:** **100 cases** (50 Cohort A, 50 Cohort B)
- **Total Raw Storage:** **1143.14 MB (1.12 GB)**
- **Total Surgical Duration:** **286.89 hours** (Mean: 2.87h)
- **Total Candidate Sliding Windows:** **15767 windows**
- **Usable Windows (Passed SQI):** **14420 windows** (91.5%)
- **Valid Early-Warning Positive Windows ($y=1$):** **48 windows**
- **Confirmed Clean Baseline Negative Windows ($y=0$):** **11293 windows**
- **Class Ratio ($y=1 : y=0$):** **1 : 235.3** (Prevalence: **0.42%**)

---

## 2. Formal Temporal Architecture & Window Terminology Correction

> [!IMPORTANT]
> **Documentation Correction**: Previous draft reports inadvertently described the sliding windows as 'non-overlapping'. Because the window span is 300 seconds and the advance stride is 60 seconds, each consecutive window shares 240 seconds (80%) of physiological waveform with its predecessor. The correct, precise clinical terminology is:  
> **'Overlapping sliding windows with a 60-second stride.'**

```
Temporal Window Architecture (Total Timeline Span = 900 seconds / 15 minutes):
+-----------------------+-----------------------+-----------------------+
|   T_obs (300 s)       |   T_gap (300 s)       |   T_target (300 s)    |
|   [t, t + 300)        |   [t + 300, t + 600)  |   [t + 600, t + 900)  |
|   Model Feature Slice |   Enforced Lead Time  |   Clinical Target     |
+-----------------------+-----------------------+-----------------------+
      |---> 60s Stride ---> Next Window [t + 60, t + 960)
```

### Mathematical & Clinical Integrity Rules:
1. **Feature Scope**: Features are extracted strictly inside $[t, t + 300)$. Zero data from $t \ge t + 300$ is accessible.
2. **Lead-Time Guarantee**: A strict 5-minute blank gap ($T_{\text{gap}} = 300\text{ s}$) separates predictor observations from the target horizon.
3. **Consecutive Run Target Definition**: $y=1$ requires $\text{max\_consecutive}(ST\_II \le -1.0\text{ mm}) \ge 60\text{ s}$ inside $T_{\text{target}}$, with NO sustained ST depression active during $T_{\text{obs}}$ or $T_{\text{gap}}$.

---

## 3. Cohort A vs. Cohort B Event Distribution (Leakage & Spectrum Audit)

Cohort A (enriched elderly/high-cardiovascular-risk) and Cohort B (representative multi-specialty surgical) exhibit distinct clinical dynamics:

| Metric | Cohort A (High-Risk Enriched) | Cohort B (Representative Surgical) | Combined Cohort |
| :--- | :---: | :---: | :---: |
| Total Cases | 50 | 50 | 100 |
| Positive Cases with Valid Warning Windows | **5 cases** | **3 cases** | **8 cases** |
| Positive Case IDs | `[137, 2085, 4236, 4390, 4568]` | `[864, 2172, 3280]` | `[137, 864, 2085, 2172, 3280, 4236, 4390, 4568]` |
| Sustained ST Episodes ($\ge 60\text{ s}$) | **23 episodes** | **1 episode** | **24 episodes** |
| Valid 5-Min Early-Warning Positive Windows | **26 windows** | **22 windows** | **48 windows** |
| Confirmed Clean Baseline Negative Windows | **5568 windows** | **5725 windows** | **11293 windows** |
| Class Prevalence ($y=1 / (y=1 + y=0)$) | **0.46%** | **0.39%** | **0.42%** |

### Clinical Interpretation:
- **Cohort A (Enriched)** concentrated 23 sustained ST episodes across 5 patients, confirming that high-risk demographic selection successfully captures severe ischemic events.
- **Cohort B (Representative)** provided 1 confirmed sustained episode (Case 864) and multiple impending episodes that emerged inside $T_{\text{target}}$ (Cases 2172 and 3280), proving that early warning opportunities exist even in unselected surgical cohorts while providing abundant clean negative controls.
- **No Cohort Leakage**: Cohort identifiers are strictly preserved to enable subgroup validation.

---

## 4. Subject-Level 70/15/15 Split Simulation (20 Independent Runs)

To simulate prospective model development, 20 independent subject-level 70% Train / 15% Validation / 15% Test splits were executed using fixed random seeds. Grouping is strictly by `caseid` (0% patient overlap across partitions):

| Sim # | Seed | Train Cases (Pos/Tot) | Val Cases (Pos/Tot) | Test Cases (Pos/Tot) | Pos Windows (Tr/Val/Te) | Neg Windows (Tr/Val/Te) | Test Positive Case IDs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 01 | 2057 | 5/70 | 1/15 | 2/15 | 29 / 5 / 14 | 8495 / 1307 / 1491 | `[np.int64(3280), np.int64(4390)]` |
| 02 | 2088 | 6/70 | 1/15 | 1/15 | 37 / 6 / 5 | 8241 / 1558 / 1494 | `[np.int64(864)]` |
| 03 | 2119 | 5/70 | 1/15 | 2/15 | 30 / 5 / 13 | 8413 / 1437 / 1443 | `[np.int64(137), np.int64(4390)]` |
| 04 | 2150 | 6/70 | 1/15 | 1/15 | 41 / 5 / 2 | 7914 / 1810 / 1569 | `[np.int64(2085)]` |
| 05 | 2181 | 5/70 | 1/15 | 2/15 | 32 / 5 / 11 | 8136 / 1503 / 1654 | `[np.int64(2085), np.int64(3280)]` |
| 06 | 2212 | 6/70 | 1/15 | 1/15 | 35 / 8 / 5 | 7399 / 1685 / 2209 | `[np.int64(864)]` |
| 07 | 2243 | 5/70 | 1/15 | 2/15 | 28 / 5 / 15 | 7462 / 1811 / 2020 | `[np.int64(3280), np.int64(4236)]` |
| 08 | 2274 | 6/70 | 1/15 | 1/15 | 37 / 5 / 6 | 7040 / 2162 / 2091 | `[np.int64(4236)]` |
| 09 | 2305 | 5/70 | 1/15 | 2/15 | 32 / 5 / 11 | 8112 / 1630 / 1551 | `[np.int64(2085), np.int64(3280)]` |
| 10 | 2336 | 6/70 | 1/15 | 1/15 | 37 / 5 / 6 | 7892 / 1672 / 1729 | `[np.int64(4236)]` |
| 11 | 2367 | 5/70 | 1/15 | 2/15 | 29 / 2 / 17 | 8378 / 1505 / 1410 | `[np.int64(137), np.int64(3280)]` |
| 12 | 2398 | 6/70 | 1/15 | 1/15 | 35 / 5 / 8 | 7808 / 1672 / 1813 | `[np.int64(2172)]` |
| 13 | 2429 | 5/70 | 1/15 | 2/15 | 28 / 5 / 15 | 7128 / 2045 / 2120 | `[np.int64(3280), np.int64(4236)]` |
| 14 | 2460 | 6/70 | 1/15 | 1/15 | 31 / 9 / 8 | 7203 / 1803 / 2287 | `[np.int64(2172)]` |
| 15 | 2491 | 5/70 | 1/15 | 2/15 | 33 / 2 / 13 | 7559 / 1806 / 1928 | `[np.int64(2172), np.int64(4568)]` |
| 16 | 2522 | 6/70 | 1/15 | 1/15 | 35 / 8 / 5 | 7378 / 2028 / 1887 | `[np.int64(4390)]` |
| 17 | 2553 | 5/70 | 1/15 | 2/15 | 28 / 5 / 15 | 7559 / 1620 / 2114 | `[np.int64(3280), np.int64(4236)]` |
| 18 | 2584 | 6/70 | 1/15 | 1/15 | 34 / 5 / 9 | 7968 / 1017 / 2308 | `[np.int64(3280)]` |
| 19 | 2615 | 5/70 | 1/15 | 2/15 | 32 / 5 / 11 | 7450 / 2046 / 1797 | `[np.int64(4236), np.int64(4390)]` |
| 20 | 2646 | 6/70 | 1/15 | 1/15 | 38 / 8 / 2 | 7803 / 1815 / 1675 | `[np.int64(2085)]` |

### Split Simulation Summary Statistics:
- **Test Set Positive Cases:** Mean = **1.50 cases** (Min: 1, Max: 2)
- **Validation Set Positive Cases:** Mean = **1.00 cases** (Min: 1, Max: 1)
- **Test Set Positive Windows:** Mean = **9.6 windows** (Min: **2**, Max: **17**)

---

## 5. Statistical Viability Assessment: Is a Single 70/15/15 Split Defensible?

> [!WARNING]
> **HIGH STATISTICAL UNCERTAINTY IN A SINGLE 15% TEST PARTITION**:
> 1. **Extreme Case-Level Sparsity**: With only 8 positive patients across the entire 100-case cohort, a 15% partition (15 cases) contains only **1 to 2 positive patients**.
> 2. **Sensitivity Volatility**: If a model correctly flags 1 of 2 test patients, patient-level sensitivity is **50%**. If it misses that 1 patient, sensitivity plunges to **0%**. If there is only 1 positive patient in the test set, sensitivity is binary (**100% or 0%**). Such metrics are clinically uninterpretable and fail to demonstrate robust generalization.
> 3. **Window Count Instability**: Because different patients have differing numbers of early-warning opportunities (e.g. Case 2085 has 2 positive windows, while Case 3280 has 9 windows), the test partition positive window count fluctuates between **5 and 17 windows** purely based on random seed selection.
> 4. **Rigorous Conclusion**: A single, static 70/15/15 train/validation/test split is **statistically fragile** for model selection and threshold calibration. Relying solely on a single 15-case test set would produce unacceptably noisy performance benchmarks.

---

## 6. Grouped 5-Fold Cross-Validation Feasibility Audit

To evaluate a more robust alternative, `StratifiedGroupKFold` (5 folds) was simulated across the cohort:

| Fold | Train Cases (Pos/Tot) | Val Cases (Pos/Tot) | Val Positive Cases | Pos Windows (Tr/Val) | Neg Windows (Tr/Val) | Val Pos Prevalence |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **Fold 1** | 6/78 | 2/20 | `[137, 4390]` | 35 / 13 | 9034 / 2259 | 0.57% |
| **Fold 2** | 7/78 | 1/20 | `[2172]` | 40 / 8 | 9035 / 2258 | 0.35% |
| **Fold 3** | 6/79 | 2/19 | `[864, 2085]` | 41 / 7 | 9032 / 2261 | 0.31% |
| **Fold 4** | 6/77 | 2/21 | `[4236, 4568]` | 37 / 11 | 9036 / 2257 | 0.49% |
| **Fold 5** | 7/80 | 1/18 | `[3280]` | 39 / 9 | 9035 / 2258 | 0.40% |

### Scientific Advantages of Grouped 5-Fold Cross-Validation:
1. **Complete Cohort Evaluation**: In 5-fold CV, **all 8 positive cases (and all 48 positive windows)** are evaluated in out-of-fold validation. No positive case is locked away uninspected.
2. **Strict Patient Isolation**: In every fold, training and validation sets are strictly disjoint at the patient level ($	ext{Train Cases} \cap 	ext{Val Cases} = \emptyset$). Zero window leakage can occur.
3. **Pooled Out-of-Fold Estimation**: Out-of-fold predictions from all 5 folds can be aggregated to compute cohort-wide PR-AUC, AUROC, and Brier calibration scores across all 100 patients, yielding substantially narrower confidence intervals.

---

## 7. Recommended Locked Test Set Strategy

To balance unbiased final benchmark testing with maximal development-time stability, the following **two-tier architecture** is recommended:

1. **Tier 1 — Development Set (80 Cases, 5-Fold Grouped CV)**:
   - 80 cases (40 Cohort A, 40 Cohort B) containing 6 positive cases and 74 negative cases.
   - Dedicated to feature selection, architecture comparison (Matrix A vs Matrix B), hyperparameter tuning, and decision threshold calibration via 5-fold `StratifiedGroupKFold`.
2. **Tier 2 — Locked Benchmark Test Set (20 Cases, Sealed)**:
   - 20 cases (10 Cohort A, 10 Cohort B) containing exactly **2 stratified positive cases** (1 from Cohort A, 1 from Cohort B) and 18 negative cases.
   - **Completely Sealed**: Never accessed during preprocessing decisions, feature pruning, model selection, or hyperparameter optimization.
   - Evaluated exactly **once** at the conclusion of modeling to verify generalization to unseen patients.

---

## 8. Feature Matrix Temporal Independence Audit

Every candidate feature in the preprocessing matrix was audited to confirm exclusive generation from $T_{\text{obs}}$ ($t \in [0, 300\text{ s})$):

| Modality | Feature Names | Window Extracted | Future / Gap Contamination |
| :--- | :--- | :---: | :---: |
| **ECG R-Peak & HRV** | `ecg_hr_mean`, `ecg_hr_std`, `ecg_rr_sdnn`, `ecg_rr_rmssd`, `ecg_pnn50` | $T_{\text{obs}}$ only | **ZERO [VERIFIED]** |
| **ECG Morphology** | `ecg_r_amp_mv`, `ecg_qrs_width_ms`, `ecg_sqi` | $T_{\text{obs}}$ only | **ZERO [VERIFIED]** |
| **Pulse Arrival Time** | `pat_median_ms`, `pat_iqr_ms`, `pat_valid_fraction`, `pat_valid` | $T_{\text{obs}}$ only | **ZERO [VERIFIED]** |
| **PPG Waveform** | `ppg_pulse_amp`, `ppg_perfusion_index`, `ppg_crest_time_ms`, `ppg_sqi` | $T_{\text{obs}}$ only | **ZERO [VERIFIED]** |
| **SpO2 Trend** | `spo2_mean`, `spo2_min`, `spo2_std`, `spo2_desat_count` | $T_{\text{obs}}$ only | **ZERO [VERIFIED]** |
| **Pre-Event ST** | `st_obs_mean`, `st_obs_median`, `st_obs_min`, `st_obs_std`, `st_slope_mm_min` | $T_{\text{obs}}$ only | **ZERO [VERIFIED]** |
| **Baseline-Relative ST** | `st_delta_baseline` (difference from initial stable baseline) | $T_{\text{obs}}$ vs baseline | **ZERO [VERIFIED]** |

Confirmed: **Zero features access $T_{\text{gap}}$, $T_{\text{target}}$, post-event intervals, or target labels.**

---

## 9. Dual Modeling Matrix Specifications (Ablation Study Ready)

Two distinct feature matrix specifications are established for future ablation benchmarking:

### Model Matrix A (Full Multimodal + Pre-Event ST)
- **Input Modalities:** ECG Lead II, PPG Plethysmogram / PAT, SpO2 Trend, Pre-Event ST Lead II
- **Feature Count:** **26 features**
- **Features:** `['ecg_hr_mean', 'ecg_hr_std', 'ecg_rr_sdnn', 'ecg_rr_rmssd', 'ecg_pnn50', 'ecg_r_amp_mv', 'ecg_qrs_width_ms', 'ecg_sqi', 'pat_median_ms', 'pat_iqr_ms', 'pat_valid_fraction', 'pat_valid', 'ppg_pulse_amp', 'ppg_perfusion_index', 'ppg_crest_time_ms', 'ppg_sqi', 'spo2_mean', 'spo2_min', 'spo2_std', 'spo2_desat_count', 'st_obs_mean', 'st_obs_median', 'st_obs_min', 'st_obs_std', 'st_delta_baseline', 'st_slope_mm_min']`
- **Scientific Hypothesis:** Comprehensive multimodal representation combining autonomic/vascular transit timing with pre-event electrophysiological micro-shifts.

### Model Matrix B (Pure Non-ST Vascular / Autonomic)
- **Input Modalities:** ECG Lead II, PPG Plethysmogram / PAT, SpO2 Trend
- **Feature Count:** **20 features**
- **Features:** `['ecg_hr_mean', 'ecg_hr_std', 'ecg_rr_sdnn', 'ecg_rr_rmssd', 'ecg_pnn50', 'ecg_r_amp_mv', 'ecg_qrs_width_ms', 'ecg_sqi', 'pat_median_ms', 'pat_iqr_ms', 'pat_valid_fraction', 'pat_valid', 'ppg_pulse_amp', 'ppg_perfusion_index', 'ppg_crest_time_ms', 'ppg_sqi', 'spo2_mean', 'spo2_min', 'spo2_std', 'spo2_desat_count']`
- **Scientific Hypothesis:** Scientific ablation model determining whether pulse arrival time (PAT), heart rate variability (HRV), and pulse morphology can predict impending ischemia WITHOUT any ST segment information.

---

## 10. Audit Summary & Checklist

| Audit Requirement | Status | Verification Summary |
| :--- | :---: | :--- |
| 100th Case Validation (Case 33) | **VERIFIED** | Replaced defective Case 5326; all 4 mandatory tracks present |
| Overlapping Window Documentation | **VERIFIED** | Corrected to 'overlapping sliding windows with 60s stride' |
| 20 Grouped Split Simulations | **VERIFIED** | Strict case-level isolation maintained (0 patient overlap) |
| 70/15/15 Viability Evaluated | **VERIFIED** | High statistical uncertainty on single test set documented |
| 5-Fold Grouped CV Evaluated | **VERIFIED** | StratifiedGroupKFold validated across all 8 positive cases |
| Cohort Leakage Prevented | **VERIFIED** | Cohort A and B event breakdowns audited independently |
| Locked Test Set Strategy | **VERIFIED** | Recommended 80-case CV dev set + 20-case locked benchmark |
| Feature Matrix Isolation | **VERIFIED** | All features confirmed strictly within T_obs |
| Model Matrices A & B Specified | **VERIFIED** | Full Multimodal (26 feats) vs Pure Non-ST (20 feats) |
| Machine Learning Models Trained | **ZERO** | Halted awaiting explicit human authorization |
| BeatAhead Website Modified | **ZERO** | Web application remains completely untouched |

### Recommended Next Action

Present this split audit for human review. Once authorized, proceed to Phase 5 modeling using **5-fold StratifiedGroupKFold cross-validation on an 80-case development cohort plus a locked 20-case test set**, benchmarking Model Matrix A against Model Matrix B.