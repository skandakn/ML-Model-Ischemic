# BeatAhead Cardiac Physiological ML Prototype: VitalDB Implementation Plan (Sprint 1)

**Document:** `reports/vitaldb_plan.md`  
**Target:** Implementation Sprint 1 — VitalDB Acquisition, Inspection, & Pre-Training Architecture  
**Date:** September 2026  
**Status:** PROPOSED PLAN (Awaiting User Review before full dataset download)  

---

## 1. Dataset Version, Source, and Access Architecture

### 1.1 Exact Dataset Version and Provenance
* **Dataset**: VitalDB (A high-fidelity multi-parameter vital signs database in surgical patients).
* **Version**: Version 1.0.1 (Data release 2022; packed `.vital` format v1.0.1).
* **Origin**: Seoul National University Hospital (SNUH), Department of Anesthesiology and Pain Medicine.
* **Open Repository Hosting**:
  * Official API & Data Portal: `https://api.vitaldb.net` / `https://vitaldb.net`
  * PhysioNet Distribution: `https://physionet.org/content/vitaldb/1.0.0/`
* **License**: **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Permits academic and commercial use, analysis, and redistribution with proper attribution.

### 1.2 Access Modality
VitalDB provides two complementary access routes:
1. **REST / HTTP Open Endpoint (`https://api.vitaldb.net`)**:
   * Metadata tables:
     * `https://api.vitaldb.net/cases` (6,388 cases, 74 clinical parameters).
     * `https://api.vitaldb.net/trks` (486,449 track records across all cases).
     * `https://api.vitaldb.net/labs` (928,448 laboratory measurement entries across 34 standard tests).
   * Binary packed case records:
     * `https://api.vitaldb.net/1.0.1/{caseid}.vital`
2. **Python Client Library (`vitaldb` v1.7.2)**:
   * Installed in the local project environment (`.venv`).
   * Provides `vitaldb.VitalFile(filepath_or_url, track_names)` and `vitaldb.load_case(caseid, track_names, interval)`.
   * **Crucial Implementation Note**: The `.vital` binary format is the authoritative time-synchronized source; all tracks in a `.vital` file share a unified hardware-synchronized timeline, preventing inter-track temporal drift.

---

## 2. Required Physiological Signals & Monitoring Tracks

To fulfill the BeatAhead multimodal pipeline requirements without incurring excessive storage overhead, we isolate the following specific tracks:

```
+----------------------------------------------------------------------------------------------------+
|                                    TARGET BIOSIGNAL TRACKS                                         |
+----------------------+---------------------------+---------------+---------------------------------+
| Modality             | Exact VitalDB Track Name  | Sampling Rate | Physiological Function          |
+----------------------+---------------------------+---------------+---------------------------------+
| 1. ECG (Waveform)    | SNUADC/ECG_II             | 500 Hz        | Primary cardiac rhythm, QRS,    |
|                      |                           |               | ST segment, HRV metrics         |
|    ECG (Lead V5)     | SNUADC/ECG_V5 (optional)  | 500 Hz        | Lateral myocardial ischemia     |
|                      |                           |               | (available in 3,390 cases)      |
+----------------------+---------------------------+---------------+---------------------------------+
| 2. PPG (Waveform)    | SNUADC/PLETH              | 500 Hz        | Peripheral microvascular volume |
|                      |                           |               | Pulse Arrival Time (PAT), crest |
|                      |                           |               | time, pulse wave amplitude      |
+----------------------+---------------------------+---------------+---------------------------------+
| 3. SpO2 (Numeric)    | Solar8000/PLETH_SPO2      | ~0.5 Hz (2s)  | Continuous arterial oxygenation |
|                      |                           |               | Desaturation episodes (<90%)    |
+----------------------+---------------------------+---------------+---------------------------------+
| 4. ST Deviation      | Solar8000/ST_II           | ~0.5 Hz (2s)  | Continuous ST segment level     |
|    (Reference)       | Solar8000/ST_V5 (partial) |               | in millivolts/mm for validation |
+----------------------+---------------------------+---------------+---------------------------------+
| 5. Hemodynamic Ref   | Solar8000/ART_MBP         | ~0.5 Hz (2s)  | Invasive Mean Arterial Pressure |
|    (Hypotension)     | (or Solar8000/NIBP_MBP)   |               | Perfusion pressure ground truth |
+----------------------+---------------------------+---------------+---------------------------------+
```

### Cohort Track Coverage
* Cases with `SNUADC/ECG_II`: **6,355 / 6,388 (99.5%)**
* Cases with `SNUADC/PLETH`: **6,157 / 6,388 (96.4%)**
* Cases with `Solar8000/PLETH_SPO2`: **6,386 / 6,388 (99.9%)**
* Cases with `Solar8000/ST_II`: **5,978 / 6,388 (93.6%)**
* **Simultaneous Intersection (`ECG_II` + `PLETH` + `PLETH_SPO2` + `ST_II`)**: **5,755 cases (90.1%)**.

---

## 3. Clinical Prediction Targets & The Ground Truth Reality

### 3.1 The Troponin I / Myocardial Injury Reality Check
> [!IMPORTANT]
> **Strict Clinical Fact**: An exhaustive audit of the public open-access VitalDB repository (`labs.csv`) confirms that **Troponin I (cTnI) and CK-MB are NOT included in the open public distribution**. The 34 public laboratory parameters consist of standard CBC, basic chemistry, arterial blood gas, and routine coagulation tests (`alb`, `alt`, `aptt`, `ast`, `bun`, `cr`, `crp`, `gfr`, `gluc`, `hb`, `hct`, `k`, `lac`, `na`, `ph`, `plt`, `ptinr`, etc.).
> 
> Studies in the medical literature evaluating "MINS" (Myocardial Injury after Noncardiac Surgery) using Seoul National University Hospital data were conducted by the hospital's clinical staff who possessed internal, institutional Electronic Medical Record (EMR) linkages that are not published openly to preserve patient de-identification.

### 3.2 How We Will Handle Myocardial Injury Labels Without "Inventing" Data
In strict accordance with the project directives (*"Do not invent missing labels or claim clinical relevance that the dataset does not actually support"*):

1. **Pipeline Label Schema (Forward Compatibility)**:
   * We write the label construction architecture to accept an external `troponin_outcomes.csv` table if institutional access or an authorized supplementary file is subsequently obtained.
   * Construction rule: A binary label `MINS_positive = 1` defined by Fourth Universal Definition of Myocardial Infarction / MINS criteria: post-operative $\text{cTnI} \ge 0.04\text{ ng/mL}$ (or assay 99th percentile URL) within 30 days post-surgery, with baseline pre-operative value subtracted.
2. **Immediate Investigational Targets Supported by Public Ground Truth**:
   For Sprint 1, we formulate two defensible, clinically grounded prediction targets natively supported by the open VitalDB signals:

   * **Target 1: Continuous Intraoperative Ischemic ST Deviation (Primary Electrophysiological Target)**
     * *Definition*: Episodes of clinically significant ST-segment depression ($\le -0.1\text{ mV} / -1.0\text{ mm}$) or elevation ($\ge +0.1\text{ mV} / +1.0\text{ mm}$) lasting $\ge 60\text{ seconds}$ on Lead II (`Solar8000/ST_II`) cross-validated by automated morphological feature extraction on high-resolution `SNUADC/ECG_II`.
     * *Clinical Relevance*: Intraoperative ST-segment depression is the classic electrophysiological sign of subendocardial myocardial ischemia induced by supply-demand mismatch during surgery.
   * **Target 2: Acute Myocardial Hypoperfusion / Severe Intraoperative Hypotension (Hemodynamic Target)**
     * *Definition*: Episodes of Mean Arterial Pressure (MAP) $< 65\text{ mmHg}$ for $\ge 5\text{ consecutive minutes}$ (or cumulative MAP $< 65\text{ mmHg}$ area under the curve), which is the validated clinical threshold for end-organ hypoperfusion and myocardial ischemic stress.
   * **Target 3: In-Hospital Mortality & High-Risk Cardiac Subgroup Stratification**
     * *Definition*: Binary in-hospital mortality (`death_inhosp == 1`) and prolonged ICU stay (`icu_days > 0`), stratified by patients with preoperative coronary artery disease (`dx` / `opname` containing CAD/CABG, $N=91$).

---

## 4. Time Relationship: Physiological Feature Windows vs. Prediction Horizon

To evaluate the predictive early-warning capacity of BeatAhead’s multimodal pipeline, we establish a sliding window early-warning architecture:

```
+----------------------------------------------------------------------------------------------------+
|                                    TEMPORAL WINDOWING SCHEMA                                       |
+----------------------------------------------------------------------------------------------------+

   [  Observation / Feature Window  ] ---- [ Lead Time / Prediction Gap ] ---- [   Target Horizon   ]
   <----------- T_obs -------------->      <------------- T_lead ------->      <------- T_target ---->
   e.g., 5 minutes (300 s)                 e.g., 5 minutes (300 s)             e.g., 5–10 min window
   - Extract Multimodal Features:          No data used from this period       Evaluate outcome:
     * ECG: HRV, ST level, QRS width       to ensure true early warning        * ST depression >= 0.1 mV
     * PPG: PAT, crest time, amplitude                                         * MAP < 65 mmHg
     * SpO2: mean, nadir, desat slope                                          (Binary Label: 0 or 1)
   --------------------------------------------------------------------------------------------------> Time (t)
   Window Step Size (Stride): 60 seconds
```

* **Observation Window ($T_{\text{obs}}$)**: 5 minutes (300 seconds). Captures sufficient cardiac cycles for stable frequency-domain HRV (VLF/LF/HF) and pulse arrival time (PAT) statistics.
* **Prediction Horizon / Lead Gap ($T_{\text{lead}}$)**: 5 to 10 minutes prior to event onset.
* **Target Event ($T_{\text{target}}$)**: Occurrence of sustained ischemic ST deviation or hemodynamic collapse.
* **Exclusion Buffer**: Any window overlapping with the active event or occurring within 5 minutes after event resolution is excluded from baseline training to prevent label leakage.

---

## 5. Subject-Level Train / Validation / Test Splitting

> [!WARNING]
> **Data Leakage Prevention**: In time-series physiological data with sliding windows, naive random window-level splitting causes catastrophic data leakage (neighboring windows from the same patient share identical baseline physiology).

* **Strict Subject-Level Partition**:
  * Splitting must be performed strictly on unique `subjectid` (or `caseid`, where each patient undergoes a distinct surgical procedure).
* **Split Ratios**:
  * **Train Set**: 70% of subjects
  * **Validation Set**: 15% of subjects (for hyperparameter tuning and early stopping)
  * **Test Set**: 15% of subjects (held out, locked until final evaluation)
* **Stratification Strategy**:
  * Stratified by surgical category (`department`: General Surgery, Thoracic, Vascular, Urology, etc.), preoperative cardiovascular risk (`preop_htn`, `preop_dm`), and event prevalence to ensure balanced representation across folds.

---

## 6. Confounders: Surgery, Anesthesia, and Vasoactive Drugs

Unlike ambulatory wearable studies in resting volunteers, surgical datasets contain profound pharmacological and procedural confounders that directly affect ECG, PPG, and SpO2:

```
+----------------------------------------------------------------------------------------------------+
|                                      CONFOUNDER TAXONOMY                                           |
+--------------------+------------------------------------+------------------------------------------+
| Confounder Category| Specific Factors in VitalDB        | Impact on Biosignals & Mitigations       |
+--------------------+------------------------------------+------------------------------------------+
| 1. Vasoactive Drugs| Phenylephrine (`intraop_phe`),     | Causes acute peripheral vasoconstriction,|
|    (Vasopressors / | Ephedrine (`intraop_eph`),         | damping PPG amplitude and shortening PAT |
|     Inotropes)     | Epinephrine (`intraop_epi`),       | without myocardial recovery.             |
|                    | Norepinephrine infusion            | *Mitigation*: Track drug bolus/infusion  |
|                    |                                    | timestamps as explicit control covariates|
+--------------------+------------------------------------+------------------------------------------+
| 2. Anesthetic      | Propofol (`intraop_ppf`),          | Profound vasodilation, blunt autonomic   |
|    Agents          | Volatile agents (Sevoflurane,      | response (suppressed HRV LF/HF power),   |
|                    | Desflurane), Remifentanil          | resting bradycardia.                     |
+--------------------+------------------------------------+------------------------------------------+
| 3. Surgical &      | Electrocautery (Bovie RF knives),  | Massive high-frequency electrical noise  |
|    Mechanical      | Laparoscopic insufflation (Pneumo- | and mechanical baseline wander;          |
|    Artifacts       | peritoneum), Patient repositioning | false ST segment spikes.                 |
|                    |                                    | *Mitigation*: Automated SQI gating.      |
+--------------------+------------------------------------+------------------------------------------+
| 4. Ventilatory     | Mechanical positive pressure       | Produces periodic respiratory variations |
|    Interactions    | ventilation (10–14 breaths/min)    | in PPG pulse amplitude (Pulse Pressure   |
|                    |                                    | Variation - PPV) and HRV.                |
+--------------------+------------------------------------+------------------------------------------+
```

---

## 7. Missing-Data Handling & Imputation Policy

1. **Signal Disconnections & Lead Dropouts**:
   * Continuous waveforms (`SNUADC/ECG_II`, `SNUADC/PLETH`) with $>10\%$ missing samples within a 5-minute analysis window are flagged as unrecoverable and the window is dropped.
   * Short transient dropouts ($<0.5\text{ s}$ / 250 samples) are interpolated using linear or cubic spline interpolation.
2. **Numeric Time-Series Gaps (`PLETH_SPO2`, `ST_II`)**:
   * Stored at 2-second nominal resolution. Missing observations within valid operational periods are forward-filled up to a maximum threshold of 10 seconds; beyond 10 seconds, the window is deemed corrupted.
3. **Preoperative Covariates**:
   * Continuous lab covariates (`preop_hb`, `preop_cr`) with missing values are imputed using median values conditioned on age, sex, and ASA physical status. Categorical history indicators (`preop_htn`, `preop_dm`) default to 0 (unrecorded/negative).

---

## 8. Signal Quality Checks (SQI) Architecture

To prevent corrupted biosignals from generating erroneous physiological features, all windows pass through a sequential quality filter:

```
Raw Biosignal Window (5 min)
   |
   v
[ Step 1: Range & Saturation Check ] 
   - ECG clipping at rail voltages (|V| > 5.0 mV)
   - PPG flatline (variance < 1e-4) or saturation at ADC bounds
   - SpO2 outside physiological limits (<50% or >100%)
   |
   +---> FAIL ---> Drop window / Flag SQI = 0
   |
   v PASS
[ Step 2: Morphological Consistency & QRS Detection ]
   - Pan-Tompkins / Hamilton QRS detector on ECG
   - Detect physiological HR: 30 bpm <= HR <= 220 bpm
   - Maximum allowable R-R jump: |RR_{i} - RR_{i-1}| < 0.6 * RR_{i-1} (unless ectopic)
   |
   +---> FAIL ---> Drop window / Flag SQI = 0
   |
   v PASS
[ Step 3: Multimodal Beat-to-Pulse Temporal Alignment ]
   - For every detected ECG R-peak, locate corresponding PPG systolic peak
   - Pulse Arrival Time (PAT) plausibility check: 100 ms <= PAT <= 450 ms
   - If <70% of heartbeats have valid corresponding PPG pulses, reject window
   |
   v PASS
[ Window Approved for Feature Extraction (SQI = 1) ]
```

---

## 9. Estimated Storage Requirements & Infrastructure Budget

| Asset | Scope | Unit Size | Total Estimated Size | Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Metadata CSVs** (`cases`, `trks`, `labs`) | Full 6,388 cases | ~10–35 MB each | **~60 MB total** | Downloaded once into `data/metadata/` |
| **Raw Binary Files** (`.vital` full archive) | Full 6,388 cases | ~16.7 MB avg | **~107 GB** | **Do NOT download in bulk** |
| **Filtered Pilot Cohort** (`.vital`) | 20 selected cases | ~16.7 MB avg | **~335 MB** | Downloaded for local pipeline validation |
| **Extracted Multi-track Arrays** (ECG, PPG, SpO2, ST) | Pilot (20 cases) | ~10 MB Parquet/NPY | **~200 MB** | Resampled & compressed locally |
| **Extracted Feature Windows** (Tabular) | Pilot (20 cases) | ~1 MB CSV/Parquet | **~20 MB** | Final XGBoost-ready training matrix |

---

## 10. Pilot-Cohort Strategy (Before Full Dataset Ingestion)

To guarantee software robustness, pipeline correctness, and zero storage exhaustion, we implement a **staged 3-phase rollout**:

### Phase 1: Pilot Cohort Selection ($N = 20\text{ cases}$)
* Criteria:
  1. Complete presence of all required tracks: `SNUADC/ECG_II`, `SNUADC/PLETH`, `Solar8000/PLETH_SPO2`, `Solar8000/ST_II`.
  2. Balanced case duration: $2\text{ to }4\text{ hours}$ (representing standard surgical lengths, avoiding 12-hour outlier files).
  3. Balanced event representation: 10 cases with recorded ST-segment deviations or hypotensive episodes; 10 stable control cases.
* Total pilot download: **~335 MB** (rapid, safe, low resource footprint).

### Phase 2: Pipeline Inspection & Quality Verification
* Run `src/data_inspection.py` on the downloaded pilot cases:
  * Validate millisecond alignment between ECG R-peaks and PPG pulses.
  * Compute missingness percentages and confirm SQI rejection rates.
  * Audit feature distributions (PAT, RMSSD, ST levels).

### Phase 3: Expansion to Full Processing (Post-Review Approval)
* Only after user review and confirmation will the automated download script process larger batches (e.g., $N=100$, then full cohort) in a streaming or batch-wise cached pipeline.

---

## 11. Sprint 1 Constraints & Guardrails

* [x] **No XGBoost training** executed during this sprint.
* [x] **No synthetic data** generated.
* [x] **No invented labels** (Troponin I absence explicitly documented; native continuous ST and hemodynamic ground truth targets established).
* [x] **No ISI calculation** performed yet.
* [x] **No modification** to any existing BeatAhead website repository.
