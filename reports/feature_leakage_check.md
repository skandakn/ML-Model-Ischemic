# BeatAhead 100-Case Dataset: Feature Leakage & Temporal Integrity Audit

**Date:** 2026-09-22 12:08:46  
**Status:** AUTOMATED AUDIT COMPLETE — PASSED ALL CONSTRAINTS

---

## 1. Temporal Boundary Enforcement Check

- **Observation Window Span ($T_{\text{obs}} = 300\text{ s}$):** [PASS]
- **Prediction Gap Span ($T_{\text{gap}} = 300\text{ s}$):** [PASS]
- **Target Horizon Span ($T_{\text{target}} = 300\text{ s}$):** [PASS]
- **Strict Temporal Monotonicity ($T_{\text{obs}} < T_{\text{gap}} < T_{\text{target}}$):** [PASS]

## 2. Event Contamination & Lead-Time Integrity Check

- **Active Sustained Event inside $T_{\text{obs}}$ for Positive Windows:** 0 windows ([PASS] Zero contamination)
- **Active Sustained Event inside $T_{\text{gap}}$ for Positive Windows:** 0 windows ([PASS] Zero violation of 5-min lead time)

## 3. Feature Matrix Independence Check

Total candidate features extracted strictly within $T_{\text{obs}}$: **26**

| Feature Name | Modality | Calculated Window | Leakage Risk Status |
| :--- | :---: | :---: | :---: |
| `ecg_hr_mean` | ECG | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `ecg_hr_std` | ECG | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `ecg_rr_sdnn` | ECG | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `ecg_rr_rmssd` | ECG | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `ecg_pnn50` | ECG | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `ecg_r_amp_mv` | ECG | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `ecg_qrs_width_ms` | ECG | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `ecg_sqi` | ECG | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `pat_median_ms` | PPG | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `pat_iqr_ms` | PPG | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `pat_valid_fraction` | PPG | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `pat_valid` | PPG | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `ppg_pulse_amp` | PPG | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `ppg_perfusion_index` | PPG | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `ppg_crest_time_ms` | PPG | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `ppg_sqi` | PPG | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `spo2_mean` | SpO2 | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `spo2_min` | SpO2 | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `spo2_std` | SpO2 | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `spo2_desat_count` | SpO2 | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `st_obs_mean` | Pre-Event ST | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `st_obs_median` | Pre-Event ST | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `st_obs_min` | Pre-Event ST | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `st_obs_std` | Pre-Event ST | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `st_delta_baseline` | Pre-Event ST | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |
| `st_slope_mm_min` | Pre-Event ST | $T_{\text{obs}}$ ($0-300\text{ s}$) | **CLEAN [NO LEAKAGE]** |

## 4. Exclusion & Separation Summary

- `confirmed_negative`: 11207 windows
- `ambiguous_buffer`: 2786 windows
- `sqi_fail_numeric_missingness`: 951 windows
- `sqi_fail_insufficient_st_samples`: 391 windows
- `post_event_recovery`: 135 windows
- `event_already_in_obs`: 98 windows
- `event_in_gap`: 60 windows
- `valid_early_warning_pos`: 48 windows