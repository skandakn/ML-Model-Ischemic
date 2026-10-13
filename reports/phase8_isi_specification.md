# Phase 8 — BeatAhead Ischemic Stress Index (ISI) Specification & Audit Report
**Document ID:** BEATAHEAD-PHASE8-ISI-SPECIFICATION  
**Execution Timestamp:** 2026-09-22 19:48:00 UTC  
**Environment:** BeatAhead Specification & Architectural Audit  
**Phase Status:** SPECIFICATION ONLY (Strictly Non-Executable & Non-Modifying)  
**Upstream Frozen Artifacts:**
- Frozen Model: `models/beatahead_phase5_model.joblib` (SHA256 verified)
- Frozen Schema: `models/feature_schema.json` (Schema v1.0.0, 26 Matrix A features)
- Frozen Model Metadata: `models/model_metadata.json`
- Frozen Calibrated Threshold: $\tau = 0.156742$
- Frozen Dataset: `data/processed/vitaldb_100_windows.pkl.gz` (100-case VitalDB surgical benchmark)

---

## 1. Executive Summary & Regulatory Boundary

Phase 8 defines the formal engineering, mathematical, and conceptual specification governing how BeatAhead's user-facing **0–100 Ischemic Stress Index (ISI)** relates to the upstream **Frozen Phase 5/6 Machine Learning Inference Engine**.

### 1.1 Invariants Upheld in Phase 8
In strict compliance with project constraints:
1. **Zero Retraining or Alteration**: The frozen XGBoost model artifact, hyperparameters, 26 Matrix A features, and calibrated decision threshold ($0.156742$) were untouched.
2. **Zero Code Modification**: No application code, UI components, API routes, or simulation routines (`calculateISI()`) were altered.
3. **Formal Decoupling ($p_{\text{model}} \neq \text{ISI}$)**: The specification explicitly establishes that the XGBoost conditional model probability ($p_{\text{model}}$) is **not** the ISI product score. Direct linear scaling ($\text{ISI} = p_{\text{model}} \times 100$) is formally rejected because it collapses 99% of normal physiology into the $0.0–1.5$ range and distorts user perception.
4. **Strict Non-Diagnostic Positioning**: The ISI is formally defined as an exploratory, illustrative product-level index for trend tracking and research demonstration. It does not constitute a clinically validated risk score, does not predict myocardial infarction, and carries strict prohibitions against diagnostic terminology.

---

## 2. Audit of Existing BeatAhead ISI Implementation

An exhaustive audit of the existing codebase (`C:\ischemic\src\lib\isi\` and related UI components) was conducted to baseline current behavior.

### 2.1 Current `calculateISI()` Routine (`src/lib/isi/scoring.ts`)
```typescript
export function calculateISI(input: ScoringInput): ISIScore {
  const { features, baseline, historicalScores, scenario, signalQuality, timestamp } = input;

  const hrvFactor = normalizeDeviation(features.hrv.deviation, -15, 15);
  const pulseFactor = normalizeDeviation(features.pulseMorphology.deviation, -10, 20);
  const spo2Factor = normalizeDeviation(features.spo2Trend.deviation, -3, 3);
  const ecgFactor = normalizeDeviation(features.ecg.deviation, -10, 25);
  const motionFactor = features.imu.artifactDetected ? 0.8 : Math.min(features.imu.motionIntensity, 0.5);

  const contributions: ISIContributions = {
    hrv: hrvFactor * 0.28,
    pulseMorphology: pulseFactor * 0.22,
    spo2Trend: spo2Factor * 0.18,
    ecg: ecgFactor * 0.2,
    motionArtifact: motionFactor * 0.12,
  };

  const rawScore =
    baseline.isi +
    contributions.hrv * 35 +
    contributions.pulseMorphology * 30 +
    contributions.spo2Trend * 25 +
    contributions.ecg * 35 +
    contributions.motionArtifact * 20;

  const scenarioModifier = getScenarioModifier(scenario, historicalScores.length);
  const adjustedScore = rawScore * scenarioModifier;
  const score = Math.round(Math.max(0, Math.min(100, adjustedScore)));
  ...
}
```

### 2.2 Audit Findings & Observations
1. **Inputs**:
   - `features`: Mocked `FeatureSet` containing HRV SDNN, PPG pulse morphology, SpO2 trend, ECG heart rate, and IMU motion intensity.
   - `baseline`: `PersonalBaseline` with default values: resting HR: 68 bpm, HRV: 51 ms, SpO2: 97%, pulse morphology: 0.85, baseline ISI: 48.
   - `historicalScores`: Array of recent numeric scores.
   - `scenario`: Simulated test state (`"normal" | "stress_event" | "recovering" | "persistent_rising" | "motion_artifact"`).
   - `signalQuality`: Single scalar percentage ($0–100$).
2. **Normalization & Weighting**:
   - Feature deviations are clamped and linearly normalized to $[0, 1]$ via `normalizeDeviation(val, min, max)`.
   - Weights: HRV (28%), Pulse Morphology (22%), SpO2 Trend (18%), ECG (20%), Motion Artifact (12%). Total $= 100\%$.
   - Scale multiplier: Weighted contributions are scaled by factors between 20 and 35 and added to `baseline.isi` ($48$).
3. **Simulation Modifiers**:
   - Scores are modulated by synthetic sinusoidal multipliers (`getScenarioModifier`) based on tick count.
4. **Current Thresholds & Labels (`src/lib/utils.ts`)**:
   - `0 - 30`: `"Lower observed trend"` (Green `#10B981`)
   - `31 - 60`: `"Intermediate observed trend"` (Amber `#F59E0B`)
   - `61 - 100`: `"Higher observed trend"` (Red `#DC2626`)
   - Risk Trend Banner: Triggers when $\text{score} > \text{baseline.isi} + 10$ ($> 58$ points).
5. **Existing UI References**:
   - **`ISIGauge.tsx`**: Radial circular gauge displaying 0–100 integer score, personal baseline (48), current deviation, trend arrow, confidence percentage, and disclaimer tooltip.
   - **`RiskTrendBanner.tsx`**: Context banner displaying whether physiology is within or above recent personal baseline.
   - **`ContributionBars.tsx`**: Horizontal progress bars showing relative influence of HRV, SpO2, pulse morphology, and motion.
   - **`Live Monitor` (`/monitor`)**: Integrates real-time gauge with rolling line chart.
   - **`Trends` (`/trends`)**: 24h, 7-day, and 30-day historical time-series graphs.
6. **Codebase Disconnect**:
   - The comment in `scoring.ts` states: *"Replace this function with a real XGBoost model inference call in production."*
   - This audit confirms that simply swapping `calculateISI()` with raw model probability would break the UI: an intraoperative ischemia probability of $0.005$ (normal surgical state) would display an ISI of $0$ or $1$, rendering the 48-baseline gauge completely unresponsive.

---

## 3. Five Conceptual Layer Definitions

To establish modularity and clinical clarity, BeatAhead defines five strictly separated layers:

### Layer A: Frozen Model Output
- **Nature**: Statistical binary classifier (XGBoost) trained on 100 surgical cases.
- **Inputs**: 26 Matrix A multimodal features computed over $T_{\text{obs}} = 300\text{ s}$.
- **Outputs**:
  - $p_{\text{model}} \in [0, 1]$: Statistical likelihood of prospective intraoperative ST depression event ($\ge 60$s at $\text{ST} \le -1.0\text{ mm}$ after a 300s buffer).
  - $\tau_{\text{frozen}} = 0.156742$: Frozen calibrated decision threshold.
  - $\text{ALERT} \in \{0, 1\}$: Binary state ($\text{ALERT} = 1 \iff p_{\text{model}} \ge 0.156742$).

### Layer B: Signal Quality Assessment (SQA)
- **Nature**: Multi-sensor reliability and integrity gating engine.
- **Components**:
  - $Q_{\text{ecg}} \in [0, 1]$: Clean QRS detection fraction.
  - $Q_{\text{ppg}} \in [0, 1]$: Waveform clipping and perfusion signal quality index.
  - $V_{\text{pat}} \in \{0, 1\}$: Biologically plausible pulse arrival transit flag (100–450 ms in $\ge 30\%$ beats).
  - $I_{\text{motion}} \in [0, 1]$: Inertial accelerometer disturbance magnitude.
- **Role**: Quality gating prevents noisy sensor data from masquerading as physiological deterioration.

### Layer C: Personal Baseline Tracking
- **Nature**: Patient-specific resting coordinate system.
- **Parameters**: Baseline heart rate ($HR_{\text{base}}$), variability ($SDNN_{\text{base}}$), transit ($PAT_{\text{base}}$), oxygenation ($SpO2_{\text{base}}$), and ST level ($ST_{\text{base}}$).
- **Role**: Quantifies relative physiological disturbance ($\Delta$). Disclaimed as non-predictive.

### Layer D: Temporal Trend & Momentum
- **Nature**: Rate of change and trajectory velocity tracker.
- **Metrics**: 5-minute regression slope ($v$), 30-minute cumulative trajectory, and discrete trend direction (`increasing`, `decreasing`, `stable`).
- **Role**: Differentiates acute transient fluctuations from sustained physiological drift.

### Layer E: ISI Product Score
- **Nature**: 0–100 composite product index.
- **Synthesis**: Combines Model Evidence ($E_{\text{model}}$), Autonomic Deviation ($D_{\text{auto}}$), Perfusion Strain ($D_{\text{perf}}$), and Trend Momentum ($M_{\text{trend}}$) anchored to personal resting baseline.
- **Role**: Provides a responsive, easily interpreted wellness and stress indicator for user interface surfaces.

---

## 4. Formal ISI Input Specification

| Input | Source | Units | Normalization | Direction | Missing Data Behavior | Quality Dependency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$p_{\text{model}}$** | XGBoost Matrix A | Probability $[0, 1]$ | Piecewise Sigmoidal | Higher $\rightarrow$ Higher ISI | Hold last valid up to 120s; fallback to heuristic | Gated if $Q_{\text{ecg}} < 0.50$ |
| **$\text{ALERT}$** | Threshold Comparator | Binary $\{0, 1\}$ | $\{0, 1\}$ at $\tau=0.156742$ | $1 \rightarrow$ Elevates ISI Floor | Return 0 if model offline | Inherits $p_{\text{model}}$ dependency |
| **$\Delta HR_{\text{base}}$** | ECG Lead II | bpm | $\frac{HR - HR_{\text{base}}}{HR_{\text{base}}} \times 100$ | Higher $\rightarrow$ Higher ISI | Assume $\Delta = 0$ | Clamped if $Q_{\text{ecg}} < 0.60$ |
| **$\Delta SDNN_{\text{base}}$** | ECG Lead II | ms | $\frac{SDNN - SDNN_{\text{base}}}{SDNN_{\text{base}}} \times 100$ | Lower $\rightarrow$ Higher ISI | Assume $\Delta = 0$ | Suppressed if ectopic $> 15\%$ |
| **$\Delta PAT_{\text{base}}$** | ECG + PPG Synchrony | ms | $\frac{PAT - PAT_{\text{base}}}{PAT_{\text{base}}} \times 100$ | Shorter $\rightarrow$ Higher ISI | Exclude from factor pool | Omitted if $V_{\text{pat}} == 0$ |
| **$\Delta SpO2_{\text{base}}$** | Pulse Oximeter | % | $\min(0, SpO2 - SpO2_{\text{base}})$ | Lower $\rightarrow$ Higher ISI | Assume $\Delta = 0$ | Clamped if $Q_{\text{ppg}} < 0.60$ |
| **$\Delta ST_{\text{obs}}$** | ECG Lead II | mm | $ST_{\text{median}} - ST_{\text{base}}$ | Depressed $\rightarrow$ Higher ISI | Fallback to non-ST heuristic | Gated if Lead II detached |
| **$Q_{\text{ecg}}$** | QRS Detector | Ratio $[0, 1]$ | Linear $[0, 1]$ | Higher $\rightarrow$ Higher Confidence | Defaults to 0.0 (Unusable) | Direct quality sensor |
| **$Q_{\text{ppg}}$** | PPG Pulse Analyzer | Ratio $[0, 1]$ | Linear $[0, 1]$ | Higher $\rightarrow$ Higher Confidence | Defaults to 0.0 (Unusable) | Direct quality sensor |
| **$V_{\text{pat}}$** | Pulse Transit Engine | Binary $\{0, 1\}$ | $\{0, 1\}$ | $1 \rightarrow$ Valid transit | Defaults to 0 (Invalid) | Synchrony dependency |
| **$I_{\text{motion}}$** | Accelerometer | a.u. / g | Clamped $[0, 1]$ | Higher $\rightarrow$ Lower Confidence | Defaults to 0.0 | Inertial sensor |
| **$T_{\text{velocity}}$** | Moving Regression | points/min | Bounded $[-5, +5]$ | Positive $\rightarrow$ Rising Trend | 0.0 during initial 5 min | Requires $\ge 5$ clean ticks |

---

## 5. Personal Baseline Architecture

1. **Initialization**:
   - Minimum **600 seconds (10 minutes)** of continuous monitoring.
   - Minimum data validity: $\ge 80\%$ of windows with $Q_{\text{overall}} \ge 0.70$.
   - System state during acquisition: `Baseline Establishing` (displays acquisition percentage; no premature scores).
2. **Adaptive Rolling Update (EMA)**:
   $$B_t = (1 - \alpha) B_{t-1} + \alpha X_t \quad (\alpha = 0.02)$$
3. **Anti-Drift Gating Rule (Safety Critical)**:
   $$\text{Update Allowed} \iff (\text{State} == \text{"Normal / Stable"}) \land (Q_{\text{overall}} \ge 0.75) \land (I_{\text{motion}} < 0.30)$$
   **Baseline updates are strictly locked during `Elevated Model Evidence` or `Elevated ISI Trend`.** This prevents chronic physiological stress or acute ischemia from shifting the baseline upward and masking pathology.
4. **Outlier Rejection**: Values outside $\pm 3.0 \times \text{IQR}$ are rejected as transient motion or sensor detachment.

---

## 6. Mathematical Prototype Specification

### 6.1 Model Evidence Mapping ($E_{\text{model}}$)
To bridge the frozen decision threshold ($\tau = 0.156742$) to the 0–100 scale:

$$E_{\text{model}}(p) = \begin{cases} 
0.50 \times \left( \frac{p}{\tau} \right)^{1.4} & p < \tau \\
0.50 + 0.50 \times \left( 1 - \exp\left( -4.0 \frac{p - \tau}{1 - \tau} \right) \right) & p \ge \tau
\end{cases}$$

- $p = 0.005$ (normal surgical baseline) $\rightarrow E_{\text{model}} \approx 0.008$
- $p = \tau = 0.156742$ (exact decision threshold) $\rightarrow E_{\text{model}} = 0.500$
- $p = 0.300 \rightarrow E_{\text{model}} \approx 0.77$
- $p = 0.600 \rightarrow E_{\text{model}} \approx 0.96$

### 6.2 Autonomic Shift Factor ($D_{\text{auto}}$)
$$f_{\text{hr}} = \text{clamp}\left( \frac{HR - HR_{\text{base}}}{25.0}, 0.0, 1.0 \right)$$
$$f_{\text{hrv}} = \text{clamp}\left( \frac{SDNN_{\text{base}} - SDNN}{0.60 \times SDNN_{\text{base}}}, 0.0, 1.0 \right)$$
$$D_{\text{auto}} = 0.50 \cdot f_{\text{hr}} + 0.50 \cdot f_{\text{hrv}}$$

### 6.3 Oxygenation & Perfusion Factor ($D_{\text{perf}}$)
$$f_{\text{spo2}} = \text{clamp}\left( \frac{SpO2_{\text{base}} - SpO2}{5.0}, 0.0, 1.0 \right)$$
$$f_{\text{pat}} = \text{clamp}\left( \frac{PAT_{\text{base}} - PAT}{50.0}, 0.0, 1.0 \right) \quad (\text{if } V_{\text{pat}} = 1, \text{ else omitted})$$
$$D_{\text{perf}} = 0.60 \cdot f_{\text{spo2}} + 0.40 \cdot f_{\text{pat}} \quad (\text{or } D_{\text{perf}} = f_{\text{spo2}} \text{ if PAT invalid})$$

### 6.4 Trend Momentum ($M_{\text{trend}}$)
$$M_{\text{trend}} = \text{clamp}\left( \frac{v}{2.0}, -0.15, +0.15 \right) \quad (v = \text{points / minute over 5 min})$$

### 6.5 Composite Synthesis & Guaranteed Bounding
$$\text{ISI}_{\text{raw}} = 40.0 + \left[ 0.45(E_{\text{model}} - 0.20) + 0.30 D_{\text{auto}} + 0.25 D_{\text{perf}} + M_{\text{trend}} \right] \times 60.0$$
$$\text{ISI} = \text{round}\left( \max\left( 0.0, \min\left( 100.0, \text{ISI}_{\text{raw}} \right) \right) \right)$$

$$\forall \text{ valid physiological vectors}, \quad 0 \le \text{ISI} \le 100$$

---

## 7. Model Alert State vs. ISI Display Score

| Attribute | Model Alert State ($\text{ALERT}$) | ISI Display Score ($\text{ISI}$) |
| :--- | :--- | :--- |
| **Output Type** | Discrete Binary Flag ($\{0, 1\}$) | Bounded Continuous Index ($[0, 100]$) |
| **Threshold** | Frozen Calibrated Value: **$0.156742$** | Low ($\le 30$), Mid ($31–60$), High ($\ge 61$) |
| **Meaning** | Machine learning forecast of prospective intraoperative ST depression event $\ge 60$s | Multi-factor physiological strain index relative to baseline |
| **UI Placement** | Research Model Signal Card (`/insights`) | Circular Gauge (`/monitor`, `/dashboard`) |

---

## 8. Signal Quality & Degraded Sensor Behavior

$$Q_{\text{overall}} = Q_{\text{ecg}} \times 0.40 + Q_{\text{ppg}} \times 0.35 + V_{\text{pat}} \times 0.15 + (1 - I_{\text{motion}}) \times 0.10$$

1. **$Q_{\text{ecg}} < 0.50$**: Suppress model inference; freeze ST & HRV terms; display `"Signal Degraded"`.
2. **$Q_{\text{ppg}} < 0.50$**: Suppress PPG morphology and PAT terms; calculate score solely from ECG and SpO2.
3. **$V_{\text{pat}} == 0$**: Exclude PAT from perfusion term gracefully without raising errors.
4. **Missing SpO2**: Retain nominal $SpO2_{\text{base}}$, reduce overall confidence by 20%.
5. **$I_{\text{motion}} > 0.70$**: Lock score buffer; display `"Motion Artifact Detected"`.
6. **$Q_{\text{overall}} < 0.35$**: System enters `Insufficient Signal Quality`; gauge displays `--`.

---

## 9. Five Product Output States

1. **`Insufficient Signal Quality`**: $Q_{\text{overall}} < 0.35$. Gauge shows `--`.
2. **`Baseline Establishing`**: Monitoring time $< 600\text{ s}$. Progress percentage shown.
3. **`Normal / Stable`**: $Q_{\text{overall}} \ge 0.50$, $p_{\text{model}} < 0.156742$, $\text{ISI} \le 60$. Green or amber gauge. Label: `"Lower/Intermediate observed trend"`.
4. **`Elevated Model Evidence`**: $p_{\text{model}} \ge 0.156742$. Explicit alert callout: `"Early-warning model evidence elevated"`.
5. **`Elevated ISI Trend`**: $\text{ISI} \ge 61$ due to cumulative autonomic/perfusion drift even if model is sub-threshold. Red gauge.

> [!CAUTION]
> **Prohibited Diagnostic Terminology:**  
> The system must NEVER state: *"Heart attack detected"*, *"Ischemia confirmed"*, *"Myocardial infarction occurring"*, or *"Coronary artery occluded"*. All language must remain strictly observational and trend-oriented.

---

## 10. AI Insights Explainability Boundaries

- **Permitted**:
  - Explaining whether the machine learning model detected physiological patterns matching surgical benchmark early-warning signatures.
  - Reporting percentage deviations from the individual's resting baseline.
  - Presenting relative feature contribution breakdowns (HRV, PAT, SpO2, ST).
  - Highlighting sensor quality or motion artifacts reducing confidence.
- **Strictly Forbidden**:
  - Formulating clinical diagnoses or ruling out pathology.
  - Recommending medication adjustments or cessation.
  - Reassuring users to ignore acute chest pain, shortness of breath, or radiating arm pain.

---

## 11. Delivery Verification Checklist

- [x] Full audit of existing BeatAhead ISI codebase completed and documented.
- [x] Conceptual separation of 5 layers established.
- [x] Formal ISI input specification table generated (12 tracked inputs).
- [x] Personal baseline architecture specified (initialization, anti-drift gating, outlier rejection).
- [x] Mathematical prototype specification established with guaranteed $[0, 100]$ bounds.
- [x] Model decision threshold ($0.156742$) strictly preserved.
- [x] Signal quality degradation rules defined.
- [x] 5-state product automaton defined with non-diagnostic terminology rules.
- [x] AI Insights explainability boundaries defined.
- [x] Deliverables created: `docs/ISI_SPECIFICATION.md`, `docs/ISI_DATA_FLOW.md`, `reports/phase8_isi_specification.md`.
- [x] Zero code, UI, API, model artifact, or dataset modifications made.
