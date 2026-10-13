# BeatAhead Cardiac Physiological ML Prototype: Dataset Research & Feasibility Report

**Target Document:** `reports/dataset_research.md`  
**Focus:** Multimodal physiological pipeline (ECG, PPG, SpO2, Motion/IMU) for cardiac physiological monitoring and ischemic stress detection.  
**Author:** BeatAhead ML & Clinical Physiological Research Team  

---

## 1. Executive Summary & Clinical Ground Truth Framework

The objective of the **BeatAhead** cardiac physiological prototype is to evaluate the feasibility of machine learning models (specifically gradient boosted decision trees such as XGBoost using engineered physiological features) for cardiac monitoring and ischemic stress detection using non-invasive, multimodal signals:
* **Electrocardiogram (ECG)**: Myocardial electrical activation and repolarization.
* **Photoplethysmogram (PPG)**: Peripheral microvascular blood volume oscillations.
* **Pulse Oximetry (SpO2)**: Arterial blood hemoglobin oxygen saturation.
* **Inertial Measurement Unit (IMU / Accelerometry)**: Motion kinematics and physical activity context.

### The "Ischemic Stress" Physiological Dilemma

A common and critical failure mode in wearable health tech literature is the conflation of different physiological "stress" states. When evaluating public databases—especially on PhysioNet—one must **never assume** that an annotated stress dataset provides ground truth for myocardial ischemia. Clinically, these states must be rigorously distinguished:

```
+----------------------------------------------------------------------------------------------------+
|                                    PHYSIOLOGICAL "STRESS" SPECTRUM                                 |
+------------------------------------+----------------------------------+----------------------------+
| Category                           | Core Mechanism                   | Biomarkers / Manifestations|
+------------------------------------+----------------------------------+----------------------------+
| 1. True Myocardial Ischemia        | Coronary perfusion failure:      | ST depression/elevation,   |
|    (Ischemic Stress)               | Oxygen supply < myocardial demand| T-wave inversion, cTnI/cTnT|
|                                    | (coronary stenosis, occlusion)   | troponin release, angina   |
+------------------------------------+----------------------------------+----------------------------+
| 2. Physical Exertion Stress        | Increased metabolic workload:    | Tachycardia, elevated SBP, |
|    (Exercise in healthy subjects)  | High cardiac output; coronary    | normal ST-T (intact reserve|
|                                    | flow increases to match demand   | high motion artifacts (IMU)|
+------------------------------------+----------------------------------+----------------------------+
| 3. Autonomic / Mental Stress       | Sympatho-adrenal surge:          | Elevated HR, decreased HRV,|
|    (e.g., TSST, cognitive tasks)   | Epinephrine release, peripheral  | increased EDA (galvanic    |
|                                    | vasoconstriction, intact reserve | skin response), normal ST  |
+------------------------------------+----------------------------------+----------------------------+
| 4. Systemic Hypoxemic Stress       | Arterial blood desaturation:     | Decreased SpO2 (<90%),     |
|    (e.g., sleep apnea, altitude)   | Systemic low PaO2, compensatory  | hypoxic tachycardia,       |
|                                    | tachycardia, normal vessels      | vascular vasoconstriction  |
+------------------------------------+----------------------------------+----------------------------+
```

**Key Finding of this Research**: No single public dataset currently contains ambulatory wrist PPG + chest ECG + continuous SpO2 + 6-DOF IMU recorded on coronary artery disease patients undergoing exercise-induced myocardial ischemia with gold-standard cardiologist annotations. 

Consequently, developing a defensible prototype requires an **evidence-based, dual-track data strategy**:
1. **Track 1 (Clinical Cardiac Injury / Hemodynamic Stress)**: Datasets containing synchronized ECG + PPG + SpO2 tied to verified clinical myocardial injury (troponin release, myocardial infarction) or annotated ST ischemia.
2. **Track 2 (Multimodal Wearable Fusion & Motion Dynamics)**: Datasets containing synchronized ECG + PPG + SpO2 + IMU in healthy subjects undergoing physical exertion to validate motion-robust feature extraction, Pulse Arrival Time (PAT), and noise gating.

---

## 2. Systematic Evaluation of Candidate Datasets

---

### Candidate 1: VitalDB (A High-Fidelity Multi-Parameter Vital Signs Database in Surgical Patients)

#### 1. Exact Dataset Name and Source
* **Name**: VitalDB (A high-fidelity multi-parameter vital signs database in surgical patients)
* **Source**: Seoul National University Hospital / VitalLab; hosted openly on PhysioNet (`https://physionet.org/content/vitaldb/1.0.0/`) and `vitaldb.net`.
* **Citation**: Lee HC, Park Y, Yoon SB, et al. *VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients.* Scientific Data 9, 279 (2022).

#### 2. Number of Subjects/Patients
* **6,388 surgical patients** undergoing non-cardiac or cardiac surgeries.

#### 3. Available Signals
* **Signals**: Up to 196 intraoperative physiological recording tracks:
  * **ECG**: Multi-lead waveforms (Lead II, Lead V5, Lead I).
  * **PPG**: Continuous arterial plethysmogram waveform (`SNUADC/PLETH` or monitor pleth).
  * **SpO2**: Continuous arterial oxygen saturation numerics (`Solar8000/PLETH_SPO2`).
  * **Hemodynamics**: Continuous invasive Arterial Blood Pressure (ABP), Central Venous Pressure (CVP), cardiac output, stroke volume variation (SVV).
  * **Motion / IMU**: **None** (patients are supine/immobilized in surgical suites).

#### 4. Sampling Rates
* ECG waveforms: **500 Hz**
* PPG waveforms: **100 Hz, 62.5 Hz, or 500 Hz** (depending on recording device track)
* ABP waveforms: **500 Hz**
* Clinical numerics (SpO2, Heart Rate, NIBP): **1–7 second intervals**

#### 5. Available Clinical Labels/Outcomes
* 73 perioperative clinical variables and 34 time-series laboratory variables.
* **Cardiac / Ischemic Outcomes**:
  * Pre- and post-operative **cardiac Troponin I (cTnI)** and **CK-MB** laboratory values.
  * Clinical diagnosis of **Postoperative Myocardial Infarction (PMI)** and **Myocardial Injury after Noncardiac Surgery (MINS)**.
  * Intraoperative adverse cardiac events: severe hypotension (MAP < 65 mmHg, prolonged duration), hypoxemia (SpO2 < 90%), severe tachycardia/bradycardia, and 30-day mortality.

#### 6. Defensibility of ML Prediction Target
* **Defensibility: HIGH (Clinical Myocardial Injury / Hemodynamic Collapse)**.
  * *Target A (Biomarker ground truth)*: Binary classification or regression of postoperative myocardial injury (elevated post-op Troponin I above 99th percentile upper reference limit) using intraoperative windowed physiological features.
  * *Target B (Acute hemodynamic instability)*: Real-time prediction of acute hypotension/hypoperfusion episodes (coronary hypoperfusion precursor) within the subsequent 5–10 minute window.
  * *Important Caveat*: The dataset does not provide second-by-second manual ST segment episode annotations drawn by a cardiologist; ST deviation must be computed algorithmically from Lead II/V5 or linked to the troponin outcome.

#### 7. Sensor Synchronization
* **ECG, PPG, SpO2 Synchronization**: **YES, millisecond-accurate**. Data was acquired using the proprietary Vital Recorder application directly interfacing hospital patient monitors (GE Tram-Rac, Philips IntelliVue, Maquet) with unified clock timestamps.
* **Motion / IMU**: Not available.

#### 8. License and Usage Restrictions
* **License**: **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
* **Access**: Open access; no credentialing or CITI certifications required. Python library `vitaldb` allows immediate programmatic streaming.

#### 9. Major Limitations
1. **Lack of Motion/IMU Data**: All data collected under anesthesia; cannot train accelerometer-based motion artifact cancellation.
2. **Confounding Anesthetic Drugs**: Vasoactive drugs (phenylephrine, norepinephrine, ephedrine) and general anesthetics alter vascular tone and autonomic heart rate variability (HRV).
3. **Immobilized Setting**: Reflects perioperative physiology rather than ambulatory wearable activity.

#### 10. Suitability for XGBoost Model
* **Suitability: EXCEPTIONAL**.
  * Highly suited for tabular feature extraction across 30-second to 5-minute sliding windows:
    * *ECG*: Mean HR, RMSSD, SDNN, spectral LF/HF ratio, ST-segment elevation/depression amplitude (Lead II and V5 at J+60ms), QRS duration.
    * *PPG*: Pulse Arrival Time (PAT = time from ECG R-peak to PPG systolic peak), pulse amplitude, crest time, augmentation index, dP/dt.
    * *SpO2*: Mean SpO2, desaturation nadir, desaturation duration.
  * Forms a pristine tabular dataset of $N > 50,000$ feature windows mapped to troponin elevation or hypotensive stress for XGBoost.

---

### Candidate 2: Pulse Transit Time PPG Dataset (PhysioNet)

#### 1. Exact Dataset Name and Source
* **Name**: Pulse Transit Time PPG Dataset (PTT PPG)
* **Source**: PhysioNet (`https://physionet.org/content/pulse-transit-time-ppg/1.1.0/`)
* **Citation**: Boccignone G, Conte D, Cuculo V, et al. *An Open-Access Database for Evaluating Pulse Transit Time and Photoplethysmogram-Based Blood Pressure Estimation.* Sensors 2020, 20(24), 7114.

#### 2. Number of Subjects/Patients
* **22 healthy subjects** (13 male, 9 female, aged 19–56, normotensive).

#### 3. Available Signals
* **Complete Multimodal Suite**:
  * **PPG**: 2 sets of 3-wavelength raw PPG signals (Green 525nm, Red 650nm, Infrared 870nm) measured at two arterial sites (wrist and index finger).
  * **ECG**: Lead I / modified chest lead.
  * **Motion / IMU**: 3-axis Accelerometer + 3-axis Gyroscope (6-DOF) co-located on the sensor housing.
  * **Biomechanical Auxiliaries**: Sensor attachment pressure and skin temperature.

#### 4. Sampling Rates
* **500 Hz uniform sampling** across ALL channels (ECG, all PPG channels, 6-DOF IMU, pressure).

#### 5. Available Clinical Labels/Outcomes
* Physical activity state annotations:
  1. **Sitting (Rest)**
  2. **Walking (4 km/h on treadmill)**
  3. **Running (8 km/h on treadmill)**
* Spot physiological ground truth:
  * Baseline and post-activity Systolic and Diastolic Blood Pressure (manual sphygmomanometer).
  * Baseline and post-activity SpO2 and Heart Rate.
  * Demographic metadata (age, sex, height, weight, BMI).

#### 6. Defensibility of ML Prediction Target
* **Defensibility: STRICTLY LIMITED TO EXERTION / PTT / MOTION (NO ISCHEMIA)**.
  * **CRITICAL CLINICAL REALITY**: All participants are **young, healthy volunteers**. There is **ZERO** myocardial ischemia, zero coronary stenosis, and zero pathological hypoxia.
  * *Valid ML Target*: (a) Physical exertion state classification (Rest vs. Moderate vs. Vigorous exertion), (b) Continuous Blood Pressure / Hemodynamic surge estimation via PAT, (c) Motion artifact corruption level detection.
  * *Invalid Claim*: Claiming this dataset detects "ischemic stress" is medically fraudulent; it reflects healthy sympathetic cardiac compensation during exercise.

#### 7. Sensor Synchronization
* **Synchronization: PERFECT**. All 19 channels were acquired through an onboard ARM Cortex-M4 microcontroller running at 180 MHz, guaranteeing simultaneous hardware acquisition within a 2 ms window at 500 Hz.

#### 8. License and Usage Restrictions
* **License**: **Open Data Commons Open Database License v1.0 (ODbL)**.
* **Access**: Fully public on PhysioNet; no credentialing needed.

#### 9. Major Limitations
1. **Zero Pathological Labels**: No myocardial ischemia, arrhythmia, or arterial desaturation.
2. **Small Cohort**: Only 22 subjects; total recording duration per subject is relatively short (~15–20 minutes total across conditions).
3. **Discrete SpO2/BP**: SpO2 and blood pressure are pre/post intermittent measurements, not continuous waveform tracks.

#### 10. Suitability for XGBoost Model
* **Suitability: HIGH (for Wearable Signal Quality & Exertion Dynamics)**.
  * Ideal for training an XGBoost pipeline on:
    * IMU-gated signal quality index (SQI) regression.
    * PAT extraction under motion.
    * Physical workload / exertion classification using combined ECG (HR, HRV) + PPG (pulse morphology) + IMU (RMS acceleration, spectral energy).

---

### Candidate 3: Long-Term ST Database (LTST DB)

#### 1. Exact Dataset Name and Source
* **Name**: The Long-Term ST Database (LTST DB)
* **Source**: PhysioNet (`https://physionet.org/content/ltstdb/1.0.0/`)
* **Citation**: Jager F, Taddei A, Moody GB, et al. *Long-term ST database: a reference for the development and evaluation of automated ischaemia detectors and for the study of the dynamics of myocardial ischaemia.* Medical & Biological Engineering & Computing 41, 172–182 (2003).

#### 2. Number of Subjects/Patients
* **80 subjects** (86 long-term ambulatory Holter ECG recordings, 21 to 24 hours each).

#### 3. Available Signals
* **ECG Only**: 2 or 3 leads of ambulatory ECG (typically modified bipolar leads: CM5, CC5, modified V1, V2, or V5).
* **PPG, SpO2, Motion / IMU**: **NONE**.

#### 4. Sampling Rates
* **250 Hz**, 12-bit resolution per channel.

#### 5. Available Clinical Labels/Outcomes
* **Gold-Standard Expert Cardiologist Annotations**:
  * Beat-by-beat QRS detection and morphology (normal, ventricular ectopic, supraventricular ectopic).
  * Annotated episodes of **transient ST-segment shifts**:
    * **Ischemic ST episodes** (significant ST deviation meeting European Society of Cardiology / AHA criteria: $\ge 0.1\text{ mV}$ depression or $\ge 0.2\text{ mV}$ elevation lasting $\ge 30\text{ s}$).
    * **Heart-rate related ST changes** (distinguishing slow rate-dependent shifts from true ischemic events).
    * **Postural / axis change ST variations** (distinguishing motion/body position shifts from ischemia).
    * Conduction abnormality / bundle branch block annotations.

#### 6. Defensibility of ML Prediction Target
* **Defensibility: GOLD STANDARD (ECG Myocardial Ischemia)**.
  * The most defensible ambulatory ECG database in medical literature for transient ischemic stress detection. It rigorously isolates true ischemic repolarization changes from benign heart-rate drifts and positional changes.
  * ML Target: Binary classification of 10s–30s ECG windows (Ischemia vs. Non-Ischemia) or detection of ischemic event onset.

#### 7. Sensor Synchronization
* **Synchronization**: N/A (unimodal ECG; leads are hardware-synchronized).

#### 8. License and Usage Restrictions
* **License**: **PhysioNet Open Access** (citation required; standard open academic reuse).

#### 9. Major Limitations
1. **Unimodal (ECG Only)**: Zero PPG, zero SpO2, zero IMU data.
2. Cannot evaluate multimodal fusion hypotheses (e.g., whether peripheral PPG pulse transit time drops prior to ECG ST-segment changes).

#### 10. Suitability for XGBoost Model
* **Suitability: EXCELLENT (for ECG Ischemia Module)**.
  * Provides an unmatched benchmark for training an XGBoost baseline on engineered ECG features:
    * ST segment level at $J+60\text{ ms}$ and $J+80\text{ ms}$ relative to the isoelectric line.
    * ST slope (horizontal, downsloping, upsloping).
    * T-wave peak amplitude, polarity, and symmetry.
    * Instantaneous heart rate and short-term HRV.

---

### Candidate 4: STAFF III Database

#### 1. Exact Dataset Name and Source
* **Name**: STAFF III Database
* **Source**: PhysioNet (`https://physionet.org/content/staffiii/1.0.0/`)
* **Citation**: Pettersson J, Carro E, Sörnmo L, et al. *The STAFF III database: ECGs recorded during prolonged balloon inflation.* Journal of Electrocardiology 48(4), 585–593 (2015).

#### 2. Number of Subjects/Patients
* **104 patients** undergoing elective percutaneous transluminal coronary angioplasty (PTCA).

#### 3. Available Signals
* **ECG Only**: Standard diagnostic **12-lead ECG** (Leads I, II, III, aVR, aVL, aVF, V1–V6).
* **PPG, SpO2, Motion / IMU**: **NONE**.

#### 4. Sampling Rates
* **1000 Hz**, uncompressed high-fidelity recording.

#### 5. Available Clinical Labels/Outcomes
* **Pristine Experimental Occlusion Ground Truth**:
  * Exact millisecond timestamps of **balloon inflation** (start of total mechanical coronary occlusion = acute transmural ischemia).
  * Exact timestamps of **balloon deflation** (reperfusion).
  * Exact anatomical occlusion site (Left Anterior Descending [LAD], Right Coronary Artery [RCA], Left Circumflex [LCx]).
  * Balloon inflation duration (average 1.5–5 minutes).
  * Clinical chest pain severity scores and angiographic stenosis severity.

#### 6. Defensibility of ML Prediction Target
* **Defensibility: ABSOLUTE GROUND TRUTH FOR ACUTE TRANSMURAL ISCHEMIA**.
  * Balloon inflation is the human clinical model of induced acute zero-flow myocardial ischemia. There is zero ambiguity regarding the onset, duration, and anatomical territory of the ischemic insult.
  * Target: Binary classification of acute ischemia vs. pre-inflation baseline; regression of ischemic duration; or anatomical vessel localization (anterior vs. inferior).

#### 7. Sensor Synchronization
* **Synchronization**: N/A (12-lead ECG is internally synchronized at 1000 Hz).

#### 8. License and Usage Restrictions
* **License**: **PhysioNet Restricted Access**.
* **Restrictions**: Requires formal PhysioNet Credentialing, completed CITI "Data or Specimens Only Research" certification, and signing a Data Use Agreement (DUA). Cannot be accessed anonymously or instantly.

#### 9. Major Limitations
1. **ECG Only**: No PPG, SpO2, or motion data.
2. **Artificial Mechanical Occlusion**: Acute balloon occlusion in a catheterization lab produces immediate, profound transmural ischemia (STEMI-like), which is physiologically distinct from subtle exertional subendocardial ischemia (angina) in ambulatory patients.
3. **Immobile Patients**: Patients are lying supine on a catheterization fluoroscopy table.

#### 10. Suitability for XGBoost Model
* **Suitability: HIGH (for High-Fidelity ECG Validation)**.
  * Enables tabular feature extraction: 12-lead ST elevation/depression, QRS width, high-frequency QRS power, T-wave amplitude.
  * Serves as an ideal positive-control validation benchmark for any ECG ischemia feature extractor.

---

### Candidate 5: MIMIC-III Waveform Database Matched Subset

#### 1. Exact Dataset Name and Source
* **Name**: MIMIC-III Waveform Database Matched Subset
* **Source**: PhysioNet (`https://physionet.org/content/mimic3wdb-matched/1.0/`)
* **Citation**: Moody B, Moody G, Villarroel M, et al. *MIMIC-III Waveform Database Matched Subset.* PhysioNet (2020).

#### 2. Number of Subjects/Patients
* **~10,282 matched ICU patients** (~22,317 waveform records linked directly to the clinical database).

#### 3. Available Signals
* **Waveforms**:
  * Multi-lead ECG (Leads II, V, aVR, etc.).
  * PPG (continuous arterial `PLETH` waveform).
  * Arterial Blood Pressure (ABP).
  * Respiration.
* **Numerics**: Periodic time series for SpO2, Heart Rate, NIBP, Respiration Rate.
* **Motion / IMU**: **NONE**.

#### 4. Sampling Rates
* Waveforms (ECG, PPG, ABP): **125 Hz** (standardized downsampled WFDB archive).
* Numerics (SpO2, HR): **1 sample per minute or 1 Hz**.

#### 5. Available Clinical Labels/Outcomes
* Linked to the full MIMIC-III Clinical Database:
  * ICD-9 clinical diagnosis codes: Acute Myocardial Infarction (`410.xx`), Ischemic Heart Disease (`411.xx`, `414.xx`).
  * Serial laboratory troponin tests (Troponin T and Troponin I) with exact lab draw timestamps.
  * In-hospital mortality, length of stay, cardiac arrest, defibrillation events.

#### 6. Defensibility of ML Prediction Target
* **Defensibility: MODERATE TO HIGH (Patient-Level Outcome, NOT Continuous Waveform Episode)**.
  * *Defensible*: Predicting subsequent acute troponin elevation or ICU mortality/shock from 1-hour multimodal physiological windows.
  * *Non-Defensible / High Risk*: Minute-by-minute continuous ischemia detection is **not pre-annotated**. Hospital discharge ICD codes span entire admissions; pinpointing the exact minute of ischemia onset requires noisy heuristics.

#### 7. Sensor Synchronization
* **Synchronization: YES**. Waveform channels (ECG, PPG, ABP) are synchronized at 125 Hz. Numerics (SpO2) are time-aligned via record timestamps.

#### 8. License and Usage Restrictions
* **License**: **Open Data Commons Open Database License (ODbL v1.0)**.
* **Restrictions**: Requires full PhysioNet credentialing, CITI Human Subjects Training, and signed DUA.

#### 9. Major Limitations
1. **Low Waveform Sampling Rate (125 Hz)**: 125 Hz ECG limits precision of fiducial point detection (R-peak jitter $\pm 8\text{ ms}$), affecting high-resolution HRV and ST slope measurements.
2. **No Motion / IMU**: Bedbound ICU patients.
3. **Massive Data Wrangling Overhead**: Hundreds of gigabytes; high proportion of disconnected leads and artifactual PPG damping.

#### 10. Suitability for XGBoost Model
* **Suitability: MODERATE TO HIGH**.
  * Suitable for windowed tabular features (PAT from ECG-to-PPG, pulse rate variability, mean SpO2, SpO2 desaturation variance) predicting acute clinical deterioration.
  * Less agile for early prototyping than VitalDB due to low sampling frequency (125 Hz) and credentialing barrier.

---

### Candidate 6: WESAD (Wearable Stress and Affect Detection)

#### 1. Exact Dataset Name and Source
* **Name**: WESAD (Wearable Stress and Affect Detection)
* **Source**: UCI Machine Learning Repository & University of Siegen
* **Citation**: Schmidt P, Reiss A, Duerichen R, et al. *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.* ACM ICMI 2018.

#### 2. Number of Subjects/Patients
* **15 healthy subjects**.

#### 3. Available Signals
* **Chest (RespiBAN)**: ECG, Electrodermal Activity (EDA), Electromyogram (EMG), Respiration, Skin Temperature, 3-axis Accelerometer.
* **Wrist (Empatica E4)**: PPG (Blood Volume Pulse - BVP), EDA, Skin Temperature, 3-axis Accelerometer.
* **SpO2**: **NONE**.

#### 4. Sampling Rates
* RespiBAN (Chest): **700 Hz** (ECG, ACC, Respiration).
* Empatica E4 (Wrist): PPG at **64 Hz**, Accelerometer at **32 Hz**, EDA/Temp at **4 Hz**.

#### 5. Available Clinical Labels/Outcomes
* Psychophysiological protocol condition labels:
  1. Baseline (20 min neutral reading)
  2. **Psychological Stress** (Trier Social Stress Test: public speaking + mental arithmetic)
  3. Amusement (funny video clips)
  4. Meditation

#### 6. Defensibility of ML Prediction Target
* **Defensibility: STRICTLY LIMITED TO AUTONOMIC PSYCHOLOGICAL STRESS**.
  * **CRITICAL CLINICAL REALITY**: WESAD records **autonomic sympathetic nervous system arousal**, NOT myocardial ischemia.
  * Subjects have normal coronary arteries and zero myocardial hypoxia. Using WESAD to train a model and calling it an "ischemic stress detector" is completely invalid.

#### 7. Sensor Synchronization
* **Synchronization**: Synchronized between RespiBAN and Empatica E4 using event markers and cross-correlation.

#### 8. License and Usage Restrictions
* **License**: Open access for academic/research purposes.

#### 9. Major Limitations
1. Zero cardiovascular pathology or ischemia.
2. No SpO2 channel.
3. Wrist PPG sampled at only 64 Hz, which severely restricts millisecond-level PAT calculation.

#### 10. Suitability for XGBoost Model
* **Suitability: POOR for Ischemic Cardiac Prototype; HIGH only for Autonomic Stress Baselines**.
  * Not recommended for the core BeatAhead cardiac ischemia objective.

---

### Candidate 7: MESA Sleep Dataset (Multi-Ethnic Study of Atherosclerosis)

#### 1. Exact Dataset Name and Source
* **Name**: MESA Sleep Dataset
* **Source**: National Sleep Research Resource (NSRR) / NIH-NHLBI (`https://sleepdata.org/datasets/mesa`)
* **Citation**: Chen X, Wang R, Zee P, et al. *Racial/Ethnic Differences in Sleep Disturbances: The Multi-Ethnic Study of Atherosclerosis (MESA).* Sleep 2015, 38(6), 877–888.

#### 2. Number of Subjects/Patients
* **2,056 multi-ethnic participants** (ages 54–93, large population cohort with high prevalence of cardiovascular risk factors: diabetes, hypertension, coronary calcification).

#### 3. Available Signals
* Full Overnight Polysomnography (PSG):
  * **ECG**: Lead I / modified thoracic lead.
  * **PPG**: Finger photoplethysmography waveform (Nonin 8000J).
  * **SpO2**: Continuous arterial pulse oximetry.
  * **Respiration**: Abdominal and thoracic inductance plethysmography bands, airflow.
  * **Motion**: Actigraphy (7-day wrist-worn accelerometer data, available as separate files).

#### 4. Sampling Rates
* ECG: **256 Hz**
* PPG: **256 Hz**
* Continuous SpO2: **1 Hz**
* Actigraphy: 30-second activity counts

#### 5. Available Clinical Labels/Outcomes
* **Continuous Sleep Phenotypes**: Exact scored timestamps of hypopnea and obstructive apnea events, oxygen desaturations ($\ge 3\%$ and $\ge 4\%$), and respiratory arousal events.
* **Longitudinal Cardiovascular Outcomes**: Tracking of incident myocardial infarction, angina, coronary revascularization, coronary artery calcium (CAC) score, and cardiovascular death.

#### 6. Defensibility of ML Prediction Target
* **Defensibility: HIGH FOR NOCTURNAL HYPOXEMIA & CARDIOVASCULAR RISK**.
  * *Target A (Continuous Hypoxic Stress)*: Predicting desaturation severity ($\Delta\text{SpO2}$) and transient vascular vasoconstriction (PPG amplitude drop) during nocturnal breathing cessation.
  * *Target B (Long-term Ischemic Risk)*: Stratifying patients with high coronary artery calcium or incident cardiovascular events using overnight autonomic/hypoxemic dynamics.
  * *Caveat*: Not an ambulatory exercise stress dataset; reflects overnight nocturnal physiology.

#### 7. Sensor Synchronization
* **Synchronization: YES**. During overnight PSG, ECG, finger PPG, and SpO2 are recorded on a single Somte PSG system with shared hardware clock. (7-day actigraphy was recorded separately).

#### 8. License and Usage Restrictions
* **License**: NSRR Data Access Agreement. Requires registration, institutional affiliation, research proposal submission, and formal approval.

#### 9. Major Limitations
1. Overnight sleep setting (no ambulatory active exercise).
2. Wrist actigraphy is separate from the PSG montage.
3. Access approval process can take 1–3 weeks.

#### 10. Suitability for XGBoost Model
* **Suitability: HIGH (for Hypoxemic-Vascular Stress Coupling)**.
  * Excellent for tabular feature extraction: Oxygen Desaturation Index (ODI), pulse wave amplitude drops, PAT variation during desaturations, and cyclic HRV variation.

---

### Candidate 8: European ST-T Database (EDB)

#### 1. Exact Dataset Name and Source
* **Name**: European ST-T Database (EDB)
* **Source**: PhysioNet (`https://physionet.org/content/edb/1.0.0/`)
* **Citation**: Taddei A, Distante G, Emdin M, et al. *The European ST-T Database: standard evaluation of computers for the analysis of ambulatory electrocardiograms.* Computers in Cardiology 1992.

#### 2. Number of Subjects/Patients
* **79 subjects** (90 annotated 2-hour excerpts).

#### 3. Available Signals
* **ECG Only**: 2-lead ambulatory ECG.
* **PPG, SpO2, Motion / IMU**: **NONE**.

#### 4. Sampling Rates
* **250 Hz**, 12-bit.

#### 5. Available Clinical Labels/Outcomes
* Cardiologist annotations of ischemic ST-segment episodes and T-wave morphology changes.

#### 6. Defensibility of ML Prediction Target
* **Defensibility: MODERATE TO HIGH (ECG Ischemia Only)**.
  * Historic benchmark for ST-T detection. However, does not isolate heart-rate related ST drift with the same rigor as LTST DB.

#### 7. Sensor Synchronization
* **Synchronization**: N/A (ECG only).

#### 8. License and Usage Restrictions
* **License**: **PhysioNet Open Access for Non-Commercial Research Only**.
* **CRITICAL RESTRICTION**: Explicitly prohibits commercial use. Any commercial testing requires purchasing the proprietary CD-ROM from the original European consortium.

#### 9. Major Limitations
1. **Commercial prohibition**.
2. Only 2-hour excerpts (short duration).
3. ECG only (no multimodal signals).

#### 10. Suitability for XGBoost Model
* **Suitability: MODERATE (Superseded by LTST DB)**. Due to the commercial restriction and shorter duration, LTST DB is strictly preferred.

---

## 3. Comprehensive Cross-Dataset Comparison Matrix

| Candidate Dataset | Modalities Available | Sampling Rates | Synchronization Quality | Clinical / Stress Target | Target Defensibility | License / Access Barrier | XGBoost Suitability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **VitalDB** *(PhysioNet / vitaldb.net)* | ECG, PPG, SpO2, ABP, Numerics *(No IMU)* | ECG: 500 Hz<br>PPG: 100–500 Hz<br>SpO2: 1–7s | **Millisecond hardware sync** across all channels | Post-op Troponin I, MINS, PMI, severe hypotension | **High**: True clinical myocardial injury & hemodynamic collapse | **CC BY 4.0**: Fully open, instant Python access | **Exceptional**: Tabular HRV, PAT, PPG morphology, SpO2 nadir |
| **Pulse Transit Time (PTT) PPG** *(PhysioNet)* | ECG, multi-site PPG, SpO2, 6-DOF IMU, Pressure | **500 Hz uniform** across all channels | **Perfect hardware sync** (ARM Cortex-M4, <2ms) | Exertion states (Rest, Walk, Run); Pre/post BP & SpO2 | **High for Exertion/Motion; ZERO for Ischemia** (Healthy cohort) | **ODbL 1.0**: Open access on PhysioNet | **High**: Feature extraction under motion, PTT/PAT, noise gating |
| **Long-Term ST (LTST DB)** *(PhysioNet)* | ECG only (2–3 leads) *(No PPG, SpO2, IMU)* | 250 Hz, 12-bit | N/A (ECG only) | Cardiologist-annotated ischemic ST episodes vs HR/postural shifts | **Gold Standard**: Definitive ambulatory myocardial ischemia | **PhysioNet Open**: Free research access | **Excellent**: Pure ECG repolarization feature vectors |
| **STAFF III Database** *(PhysioNet)* | 12-lead ECG only *(No PPG, SpO2, IMU)* | 1000 Hz diagnostic | N/A (ECG only) | PTCA balloon inflation (acute transmural occlusion), vessel site | **Gold Standard**: Controlled zero-flow acute ischemia | **Restricted**: Requires CITI training, DUA & credentialing | **High**: 12-lead ST vector magnitude, QRS power |
| **MIMIC-III Waveform Matched** *(PhysioNet)* | ECG, PPG, SpO2, ABP *(No IMU)* | Waveforms: 125 Hz<br>SpO2: 1/min | Verified waveform synchronization | ICD-9 Acute MI, Troponin T/I, ICU mortality | **Moderate-High**: In-hospital cardiac outcomes (noisy timing) | **ODbL 1.0 / Restricted**: Credentialing & DUA required | **Moderate**: Large cohort, but 125 Hz downsampling limits PAT |
| **WESAD** *(UCI / Siegen)* | ECG, PPG, EDA, EMG, ACC *(No SpO2)* | Chest: 700 Hz<br>Wrist PPG: 64 Hz | Synchronized via cross-correlation | Psychological stress (TSST), Amusement, Meditation | **Invalid for Ischemia**: Measures autonomic sympathetic tone | **Open Academic**: Free download | **Poor for Cardiac Ischemia**; High for affective stress |
| **MESA Sleep** *(NSRR / NIH)* | ECG, PPG, SpO2, Respiration, Actigraphy | ECG/PPG: 256 Hz<br>SpO2: 1 Hz | Hardware synchronized on Somte PSG | Apnea/hypopnea desaturations, incident MI, CAC score | **High**: Hypoxic burden & long-term ischemic risk | **NSRR DUA**: Requires formal research application | **High**: Nocturnal hypoxic stress & vascular coupling |
| **European ST-T (EDB)** *(PhysioNet)* | ECG only (2 leads) *(No PPG, SpO2, IMU)* | 250 Hz, 12-bit | N/A (ECG only) | Ischemic ST and T-wave changes (cardiologist annotated) | **Moderate-High**: Classic ECG ischemia | **Restricted**: **No commercial use permitted** | **Moderate**: Superseded by LTST DB |

---

## 4. Recommended Shortlist & Strategic Trade-Offs

To construct a scientifically rigorous, defensible prototype for BeatAhead without falling into the trap of inventing missing labels or conflating healthy exercise with myocardial ischemia, we recommend a **shortlist of three complementary datasets**:

```
+----------------------------------------------------------------------------------------------------+
|                             RECOMMENDED BEATAHEAD DATASET PORTFOLIO                                |
+------------------------------------+----------------------------------+----------------------------+
| Role in Prototype Pipeline         | Recommended Candidate Dataset    | Primary Value Proposition  |
+------------------------------------+----------------------------------+----------------------------+
| Track 1: Multimodal Signals &      | Primary: VitalDB                 | Synchronized ECG+PPG+SpO2  |
|          Clinical Cardiac Outcomes |                                  | with verified Troponin /   |
|                                    |                                  | MINS outcomes (CC BY 4.0)  |
+------------------------------------+----------------------------------+----------------------------+
| Track 2: Wearable Multimodal       | Primary: Pulse Transit Time PPG  | Hardware-synchronized 500Hz|
|          Motion & PTT Engineering  |          (PTT PPG)               | ECG+PPG+SpO2+6-DOF IMU     |
|                                    |                                  | under exercise exertion    |
+------------------------------------+----------------------------------+----------------------------+
| Track 3: Diagnostic ECG Ischemia   | Primary: Long-Term ST Database   | Definitive gold-standard   |
|          Algorithm Positive Control|          (LTST DB)               | cardiologist annotations   |
|                                    |                                  | of transient ST ischemia   |
+------------------------------------+----------------------------------+----------------------------+
```

---

### Candidate A (Primary Multimodal Clinical Target): **VitalDB**

* **Why it is Recommended**:
  * It is the **only open-access dataset** that simultaneously provides high-resolution ECG (500 Hz), continuous finger PPG (100–500 Hz), and continuous SpO2, coupled with **verified laboratory cardiac biomarkers (Troponin I)** and acute myocardial infarction / MINS outcomes.
  * It eliminates the barrier of PhysioNet credentialing delays because it is released under **CC BY 4.0**.
  * The `vitaldb` Python API enables frictionless pipeline development and rapid windowed feature extraction.
* **Trade-Offs & Limitations**:
  * **No Motion/IMU**: Data was recorded in operating suites under anesthesia. You cannot evaluate accelerometer noise gating or ambulatory motion artifacts using VitalDB alone.
  * **Medication Effects**: Anesthetic agents and vasopressors modulate vascular compliance and heart rate dynamics.
* **Role in BeatAhead**: Serves as the primary ground truth for training XGBoost models on multimodal physiological coupling (ECG-PPG Pulse Arrival Time, pulse wave morphology, and SpO2 variations) to predict myocardial injury and acute hemodynamic collapse.

---

### Candidate B (Primary Wearable Hardware & Motion Benchmark): **Pulse Transit Time (PTT) PPG Dataset**

* **Why it is Recommended**:
  * Provides the exact sensor configuration proposed for the BeatAhead wearable hardware: **ECG + multi-wavelength PPG + SpO2 + 6-DOF IMU (Accelerometer and Gyroscope)**.
  * All signals are sampled at a pristine **500 Hz uniform rate** with millisecond hardware synchronization.
  * Captures realistic motion artifacts across three distinct exercise exertion intensities: sitting, walking (4 km/h), and running (8 km/h).
* **Trade-Offs & Limitations**:
  * **ZERO Ischemia**: Subjects are young, healthy individuals. It **cannot** be used to claim ischemia detection.
  * Pre- and post-activity SpO2 and blood pressure are discrete measurements rather than continuous waveform tracks.
* **Role in BeatAhead**: Serves as the benchmark for:
  1. IMU-based motion artifact detection and signal quality indexing (SQI).
  2. Pulse Arrival Time (PAT) extraction algorithms under active locomotion.
  3. Classifying exercise-induced metabolic workload and physical exertion.

---

### Candidate C (Primary ECG Ischemia Gold-Standard Positive Control): **Long-Term ST Database (LTST DB)**

* **Why it is Recommended**:
  * Unrivaled clinical validity for ambulatory ECG ischemia.
  * 86 full 24-hour recordings with meticulously adjudicated ischemic ST episodes, differentiating true ischemia from heart-rate related ST drift and positional body shifts.
  * Fully open access on PhysioNet without commercial restrictions (unlike the European ST-T database).
* **Trade-Offs & Limitations**:
  * **Unimodal**: Zero PPG, SpO2, or IMU channels.
* **Role in BeatAhead**: Serves as the independent positive-control benchmark to validate BeatAhead's ECG feature engineering sub-module (ST amplitude, ST slope, T-wave alternans, HRV). Proves that the ML pipeline can accurately detect true ischemic repolarization shifts on standard clinical data before fusing with peripheral sensors.

---

## 5. Engineered Physiological Feature Blueprint for XGBoost

For the BeatAhead prototype, XGBoost will consume tabular feature vectors extracted across sliding analysis windows (e.g., 30-second windows with 50% overlap). The feature engineering blueprint across the shortlisted datasets is structured as follows:

```
+----------------------------------------------------------------------------------------------------+
|                                  ENGINEERED FEATURE TAXONOMY                                       |
+----------------------+-----------------------------------------------------------------------------+
| Domain               | Specific Engineered Features for XGBoost Matrix                             |
+----------------------+-----------------------------------------------------------------------------+
| 1. ECG Repolarization| - ST-segment deviation amplitude at J+60ms and J+80ms (mV)                  |
|    & Morphology      | - ST slope angle (degrees / classification: horizontal, down, up)          |
|    (Ischemia Core)   | - T-wave peak amplitude, polarity, and T-wave asymmetry index               |
|                      | - QRS duration (ms) and QRS energy in 150–250 Hz band (fragmentation)       |
|                      | - QT and QTc (Bazett-corrected) intervals                                   |
+----------------------+-----------------------------------------------------------------------------+
| 2. Autonomic / HRV   | - Time-Domain: Mean HR, SDNN, RMSSD, pNN50                                  |
|    (Autonomic Tone)  | - Frequency-Domain: VLF, LF (0.04–0.15 Hz), HF (0.15–0.40 Hz), LF/HF ratio  |
|                      | - Non-Linear: Sample Entropy, Poincaré SD1, SD2, SD1/SD2 ratio              |
+----------------------+-----------------------------------------------------------------------------+
| 3. Vascular & Pulse  | - Pulse Arrival Time (PAT): Time from ECG R-peak to PPG systolic peak (ms)  |
|    Dynamics (PPG)    | - Pulse Transit Time to foot: Time from ECG R-peak to PPG foot (ms)         |
|                      | - PPG Crest Time (time from foot to systolic peak)                          |
|                      | - Augmentation Index proxy (reflection height / primary peak height)        |
|                      | - Pulse Area Ratio and dP/dt maximum systolic upslope                       |
|                      | - Perfusion Index (AC amplitude / DC baseline)                              |
+----------------------+-----------------------------------------------------------------------------+
| 4. Oximetry Dynamics | - Mean SpO2 and minimum SpO2 nadir in window                                |
|    (SpO2)            | - Oxygen Desaturation Index (number of drops >= 3% or 4% per hour)          |
|                      | - Desaturation Area / Hypoxic Burden (%·min below 90% threshold)            |
|                      | - SpO2 variance and recovery slope following nadir                          |
+----------------------+-----------------------------------------------------------------------------+
| 5. Motion Kinematics | - 3-axis Accelerometer Vector Magnitude: SVM = sqrt(Ax^2 + Ay^2 + Az^2)     |
|    & Signal Quality  | - Movement Intensity: Root Mean Square (RMS) of jerk / bandpassed ACC       |
|    (IMU Gating)      | - Dominant movement cadence / frequency (step rate via FFT)                 |
|                      | - Signal Quality Index (SQI): PPG skewness, kurtosis, template correlation  |
+----------------------+-----------------------------------------------------------------------------+
```

---

## 6. Synthesis & Next Steps for BeatAhead

1. **Do Not Claim Wearable Ischemia on Healthy Volunteer Data**:
   * When using the **PTT PPG** or **WESAD** datasets, define the prediction target explicitly as **Physical Workload / Exercise Exertion State** or **Autonomic Arousal**. Do not claim these models predict myocardial ischemia.
2. **Train the Cardiac Injury / Hemodynamic Model on VitalDB**:
   * Download a curated cohort of surgical cases from VitalDB with synchronized ECG, PPG, and SpO2 tracks along with post-operative Troponin I levels.
   * Extract the engineered features above to build an XGBoost model predicting myocardial injury (cTnI elevation) and acute hemodynamic decompensation.
3. **Validate ECG Ischemic Sensitivity on LTST DB**:
   * Validate the ECG feature extractor on the LTST DB to establish sensitivity and specificity against adjudicated ischemic ST episodes.
4. **Develop IMU Noise Gating on PTT PPG**:
   * Use the PTT PPG dataset to calibrate an XGBoost Signal Quality Index (SQI) model that detects when motion artifacts corrupt PPG/ECG signals during walking and running, preventing false-positive alarm triggers.

This staged approach ensures that the BeatAhead prototype is scientifically unimpeachable, respects medical reality, and establishes a robust multimodal machine learning architecture.
