# BeatAhead Phase 14: Existing Website Machine Learning Integration Report

**Execution Date:** 2026-09-23  
**Target Repository:** `C:\ischemic` (Branch: `staging-ml-integration`)  
**ML Project Repository:** `c:\ML Model - Ischemic`  
**Live Production Target:** `https://prevention-of-heart-attack-txdb.vercel.app/`  
**GitHub Repository:** `https://github.com/skandakn/prevention-of-heart-attack-github`  
**Final Status:** **`INTEGRATION VERIFIED`**

---

## 1. Executive Summary

Phase 14 successfully integrated the validated, frozen BeatAhead Phase 5 XGBoost machine learning model and 26-feature Matrix A schema into the **EXISTING BeatAhead website**. 

The integration strictly followed all architectural, scientific, and UI constraints:
- **Zero Website Rebuilding**: The existing Next.js App Router application was preserved in full. No second dashboard, alternative UI layout, or navigation redesign was created.
- **Zero Model Alteration**: The frozen XGBoost model artifact (`models/beatahead_phase5_model.joblib`), 26-feature schema (`models/feature_schema.json`), and operating decision threshold (`0.156742`) remain 100% bit-for-bit identical to the Phase 5/6 freeze.
- **Preserved Simulated Data Mode**: Simulated physiological waveforms (ECG, PPG, SpO2, IMU) remain active for interactive demonstration. The UI prominently displays `RESEARCH MODEL — SIMULATED INPUT` across all pages.
- **Strict Decoupling of ML Probability and Product Score**: The XGBoost probability ($p_{\text{model}} \in [0, 1]$) is strictly fed into the audited Phase 8/9 composite Ischemic Stress Index (ISI) engine ($0 \le \text{ISI} \le 100$) rather than improperly equating $\text{ISI} = p \times 100$.
- **Complete Test & Build Verification**: 72/72 automated regression tests passed (including the new Phase 14 critical pipeline test suite), TypeScript static verification passed with 0 errors, and the Next.js production build succeeded across all 38 application routes.

---

## 2. Existing Website Architecture

- **Framework**: Next.js 15.1.0 with React 19 and TailwindCSS.
- **Routing Paradigm**: App Router with 38 static and dynamic routes.
- **Core User Pages**:
  - `/dashboard`: Main live physiological monitoring dashboard with gauge, trends, banners, and statistics.
  - `/monitor`: Live waveform viewer (PPG, ECG, SpO2, IMU) with signal processing pipeline.
  - `/signals`: Physiological feature analysis table and baseline comparison.
  - `/trends`: Long-term historical trend charts with time-window filtering.
  - `/insights`: AI Insights explainability card and physiological component summaries.
  - `/clinician`: Clinical decision support prototype with patient cards and JSON report export.
- **Authentication**: Clerk middleware (`src/middleware.ts`).
- **Subscription / Billing**: Razorpay integration (`src/lib/subscription/`).
- **Specialized AI Agents**: Exotel and ElevenLabs voice calling agents (`src/server/voice-service.ts`, `src/app/api/voice/`).

---

## 3. Existing Simulation & ML Service Architecture

Prior to Phase 14:
1. `src/lib/isi/simulation.ts` and `SimulationContext.tsx` generated simulated physiological samples (`heartRate`, `hrv`, `spo2`, `ppg`, `ecg`, `imu`, `signalQuality`).
2. Client-side scoring in `src/lib/isi/scoring.ts` used an inline mathematical heuristic (`getSimulatedModelProbability`) to approximate model behavior in the browser.
3. Server route `/api/ml/predict` existed to accept 26 Matrix A features, but was only called from an isolated evaluation card on `/insights`.

---

## 4. Phase 14 Integration Architecture

The complete end-to-end production data flow was established:

```
Browser Client (SimulationContext.tsx)
  │ (Extracts 26 Matrix A features from simulated physiological waveforms)
  ▼
Next.js Server API (POST /api/ml/predict)
  │ (Validates schema, types, finite bounds; sanitizes server paths)
  ▼
Upstream ML Service (Cloud Run or Local Daemon / CLI Fallback)
  │ (Loads frozen beatahead_phase5_model.joblib; SHA-256 verified)
  ▼
Frozen XGBoost Model
  │ (Evaluates 26 features -> p_model ∈ [0, 1])
  ▼
Next.js Server Gateway
  │ (Returns standardized probability, threshold 0.156742, alert status)
  ▼
Client-Side ISI Engine (src/lib/isi/engine.ts)
  │ (Computes E_model(p), D_auto, D_perf, M_trend, Gated Anti-Drift Baseline)
  ▼
Composite ISI (0–100) & Product State Machine
  │ (Normal/Stable vs Elevated Model Evidence vs Elevated Trend vs Insufficient Quality)
  ▼
BeatAhead UI Components
  ├── DashboardStats (p_model badge, threshold, alert status)
  ├── ISIGauge (Model Evidence %, Model State NORMAL/ALERT)
  ├── RiskTrendBanner (Early-warning advisory)
  ├── Signal Processing Pipeline (/monitor)
  ├── 26-Feature Matrix A Vector Table (/signals)
  ├── AI Insights Explainability Card (/insights)
  └── Clinician Export Report with Frozen Model Provenance (/clinician)
```

---

## 5. Files Changed & Added

| File | Action | Purpose |
| :--- | :---: | :--- |
| `src/lib/isi/matrix_a.ts` | **NEW** | Generates the exact 26-feature Matrix A vector from simulated physiological waveforms matching `models/feature_schema.json` bounds and canonical order. |
| `src/lib/isi/scoring.ts` | **MODIFIED** | Updated `ScoringInput` to accept real `modelProbability` from `/api/ml/predict`, decoupling model probability from UI heuristics. |
| `src/lib/simulation/SimulationContext.tsx` | **MODIFIED** | Connects simulation ticks and scenario switches to `/api/ml/predict` asynchronously; maintains model state and passes real model probability into the ISI engine. |
| `src/components/dashboard/DashboardStats.tsx` | **MODIFIED** | Displays live ML Model status card (`p_model`, operating threshold $\tau = 15.67\%$, `NORMAL` vs `ALERT`). |
| `src/app/monitor/page.tsx` | **MODIFIED** | Embeds Frozen XGBoost Matrix A into the Signal Processing Pipeline; adds early-warning alert banner when threshold is exceeded. |
| `src/app/signals/page.tsx` | **MODIFIED** | Adds dedicated 26-feature Matrix A inspection card displaying all 26 canonical features, physiological ranges, and active values. |
| `src/app/clinician/page.tsx` | **MODIFIED** | Embeds frozen model provenance (`BeatAhead Phase 5 XGBoost`, threshold `0.156742`, artifact hash) in the JSON export report and UI banner. |
| `tests/test_website_ml_integration.py` | **NEW** | Automated regression test verifying end-to-end data flow from simulated inputs to frozen model inference and dashboard outputs. |

---

## 6. Feature Mapping: Simulated Waves to 26 Matrix A Features

All 26 features strictly adhere to `models/feature_schema.json`:

| Feature # | Canonical Feature Name | Modality | Physiological Source | Bounds |
| :---: | :--- | :---: | :--- | :---: |
| 1 | `ecg_hr_mean` | ECG | `sample.heartRate` | [20.0, 300.0] |
| 2 | `ecg_hr_std` | ECG | Heart rate std dev (autonomic tone) | [0.0, 150.0] |
| 3 | `ecg_rr_sdnn` | ECG | `sample.hrv` (SDNN in ms) | [0.0, 1000.0] |
| 4 | `ecg_rr_rmssd` | ECG | Estimated RMSSD from HRV | [0.0, 1000.0] |
| 5 | `ecg_pnn50` | ECG | Percentage RR differences > 50 ms | [0.0, 100.0] |
| 6 | `ecg_r_amp_mv` | ECG | R-wave peak amplitude (mV) | [-2.0, 15.0] |
| 7 | `ecg_qrs_width_ms` | ECG | QRS complex duration (ms) | [30.0, 300.0] |
| 8 | `ecg_sqi` | ECG | ECG signal quality ratio | [0.0, 1.0] |
| 9 | `pat_median_ms` | ECG+PPG | Median Pulse Arrival Time delay | [50.0, 600.0] |
| 10 | `pat_iqr_ms` | ECG+PPG | PAT Interquartile Range | [0.0, 500.0] |
| 11 | `pat_valid_fraction` | ECG+PPG | Plausible PAT cycle fraction | [0.0, 1.0] |
| 12 | `pat_valid` | ECG+PPG | Binary flag (valid fraction >= 30%) | [0, 1] |
| 13 | `ppg_pulse_amp` | PPG | Peak-to-trough pulse amplitude | [0.0, 50000.0] |
| 14 | `ppg_perfusion_index`| PPG | AC/DC perfusion ratio (%) | [0.0, 10000.0] |
| 15 | `ppg_crest_time_ms` | PPG | Pulse foot to systolic peak time | [20.0, 500.0] |
| 16 | `ppg_sqi` | PPG | PPG signal quality ratio | [0.0, 1.0] |
| 17 | `spo2_mean` | SpO2 | Mean oxygen saturation (%) | [40.0, 100.0] |
| 18 | `spo2_min` | SpO2 | Nadir oxygen saturation (%) | [30.0, 100.0] |
| 19 | `spo2_std` | SpO2 | Saturation trend std dev (%) | [0.0, 40.0] |
| 20 | `spo2_desat_count` | SpO2 | Duration with SpO2 <= 90% (seconds) | [0, 300] |
| 21 | `st_obs_mean` | ST_II | Mean ST level in Lead II (mm) | [-15.0, 15.0] |
| 22 | `st_obs_median` | ST_II | Median ST level in Lead II (mm) | [-15.0, 15.0] |
| 23 | `st_obs_min` | ST_II | Minimum ST level in Lead II (mm) | [-15.0, 15.0] |
| 24 | `st_obs_std` | ST_II | ST level std dev in Lead II (mm) | [0.0, 15.0] |
| 25 | `st_delta_baseline` | ST_II | ST shift from baseline (mm) | [-15.0, 15.0] |
| 26 | `st_slope_mm_min` | ST_II | Linear regression ST slope (mm/min)| [-20.0, 20.0] |

---

## 7. Model Output & ISI Engine Integration

- **Model Operating Threshold**: Exactly `0.156742` (derived from $F_1$ optimization during Phase 5).
- **Model Output**: $p_{\text{model}} \in [0, 1]$.
- **Piecewise Sigmoidal Transformation**:
  $$E_{\text{model}}(p) = \begin{cases} 0.5 \times \left(\frac{p}{\tau}\right)^{\gamma_1} & \text{if } p < \tau \\ 1 - 0.5 \times \left(\frac{1 - p}{1 - \tau}\right)^{\gamma_2} & \text{if } p \ge \tau \end{cases}$$
  Where $\tau = 0.156742$, $\gamma_1 = 1.4$, $\gamma_2 = 4.0$.
  At $p = \tau = 0.156742$, $E_{\text{model}} \equiv 0.500000$ exactly.
- **Product Score Composition**:
  $$\text{ISI} = \text{clamp}_{[0, 100]}\left(40.0 + 60.0 \times \left(0.45 E_{\text{model}} + 0.30 D_{\text{auto}} + 0.25 D_{\text{perf}} + M_{\text{trend}}\right)\right)$$
- **State Machine Integration**:
  - `Normal / Stable`: $p_{\text{model}} < 0.156742$, clean signals, stable trajectory ($\text{ISI} \approx 35–45$).
  - `Elevated Model Evidence`: $p_{\text{model}} \ge 0.156742$ ($E_{\text{model}} \ge 0.50$), triggering early-warning alert badge ($\text{ISI} \approx 60–80$).
  - `Elevated ISI Trend`: Rising multi-source trajectory without single-threshold breach.
  - `Insufficient Signal Quality`: Overall SQI $< 0.35$ (motion artifact scenario).
  - `Baseline Establishing`: Monitoring time $< 600\text{ s}$.

---

## 8. Security & Privacy Audit

- **Zero Client-Side Model Exposure**:
  - Confirmed 0 `.joblib`, `.pkl`, `.pt`, `.onnx`, or `.bin` files in `C:\ischemic\public\`.
  - Confirmed 0 model binaries in `.next/` client bundles or JavaScript chunks.
- **Server Gateway Boundary**:
  - The browser client never calls the Cloud Run service directly.
  - All ML inference passes through the Next.js server route `/api/ml/predict`.
- **Secret Protection**:
  - `ML_SERVICE_URL` and `ML_SERVICE_AUTH_TOKEN` have no `NEXT_PUBLIC_` prefix and remain server-only.
- **Error Sanitization**:
  - In the event of an upstream service failure, `/api/ml/predict` returns a generic 503 error without leaking filesystem paths, usernames, or Python tracebacks.

---

## 9. Comprehensive Verification & Build Results

### A. Python Regression Test Suite (72 / 72 Tests Passed)
```
Ran 72 tests in 8.445s
OK
- tests/test_inference.py: 11 / 11 passed
- tests/test_staging_integration.py: 14 / 14 passed
- tests/test_isi_engine.py: 18 / 18 passed
- tests/test_end_to_end_validation.py: 10 / 10 passed
- tests/test_container_service.py: 13 / 13 passed
- tests/test_website_ml_integration.py: 6 / 6 passed
```

### B. TypeScript Static Verification
```
node node_modules\typescript\bin\tsc --noEmit
Exit Code: 0 (0 errors)
```

### C. Next.js Production Build
```
npm.cmd run build
✓ Compiled successfully in 43s
✓ Linting and checking validity of types ...
✓ Generating static pages (38/38)
Exit Code: 0 across all 38 application routes
```

---

## 10. Simulated-Input Demonstration Behavior

When running in simulated demonstration mode:
1. **Normal Scenario**:
   - Simulated vitals remain normative ($HR \approx 68\text{ bpm}$, $SDNN \approx 48\text{ ms}$, $ST \approx +0.08\text{ mm}$).
   - 26-feature Matrix A vector produces $p_{\text{model}} \approx 0.005–0.030 < 0.156742$.
   - ISI gauge reads $\approx 35–42$ ("Optimal / Low Risk"), state displays `NORMAL (< 0.156742)`.
2. **Stress Event Scenario**:
   - Acute ischemic profile triggers progressive ST sagging ($ST \rightarrow -1.65\text{ mm}$), tachycardia ($HR \rightarrow 94\text{ bpm}$), and PAT delay ($PAT \rightarrow 285\text{ ms}$).
   - 26-feature Matrix A vector produces $p_{\text{model}} \approx 0.32–0.45 \ge 0.156742$.
   - ISI gauge rises to $\approx 68–82$, state switches to `ALERT (≥ 0.156742)`, triggering the early-warning advisory banner.
3. **Motion Artifact Scenario**:
   - Severe IMU motion degrades SQI ($Q_{\text{ecg}} < 0.50$, $Q_{\text{ppg}} < 0.50$).
   - Signal quality gate enforces `Insufficient Signal Quality`, preventing false alarm precision.

---

## 11. Remaining Production Limitations

1. **Simulated Input Prototype**: The website currently runs on simulated physiological streaming for demonstration; real wearable hardware streaming (via Bluetooth LE / WebSocket) requires physical wearable sensor drivers.
2. **Cloud Run Provisioning**: The containerized ML service is fully packaged and ready (`Dockerfile`, `requirements.txt`), but live deployment to Google Cloud Run remains blocked pending external GCP CLI setup and credentials (as documented in Phase 13).
3. **Investigational Status**: BeatAhead is an investigational research prototype for surgical/clinical monitoring early-warning trend analysis and is not FDA approved for diagnostic use.
