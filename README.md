# BeatAhead: Real-Time Intraoperative Ischemic Heart Disease Early Warning System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-v2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-72%20Passed-brightgreen.svg)]()

BeatAhead is an end-to-end machine learning and clinical decision-support system designed for intraoperative myocardial ischemia monitoring and prospective early warning. Developed and validated against high-resolution surgical biosignals from the **VitalDB** open clinical database, BeatAhead combines an electrophysiologically grounded XGBoost classifier with an Intraoperative Ischemia Severity Index (ISI) rule engine.

---

## Key Highlights

- **Predictive Lead Time**: Issues early warnings with a strict **5-minute lead time** prior to sustained ischemic ST-segment depression episodes ($\ge 60\text{s}$ contiguous at $ST \le -1.0\text{ mm}$).
- **Multimodal Biosignal Fusion (Matrix A)**: Integrates 26 features spanning:
  - **ECG Morphometrics & HRV**: Heart rate mean/std, SDNN, RMSSD, pNN50, R-peak amplitude, QRS duration, ECG Signal Quality Index (SQI).
  - **Vascular & Hemodynamics (PPG & PAT)**: Pulse Arrival Time (PAT median, IQR, validity fraction), pulse crest time, perfusion index, PPG SQI.
  - **Oximetry**: $\text{SpO}_2$ mean, min, std, and desaturation event counts.
  - **Electrophysiological ST Dynamics**: Mean, median, minimum, standard deviation, delta from surgical baseline, and ST slope.
- **Calibrated Decision Boundary**: Frozen Phase 5 model tuned with an optimal decision threshold ($0.156742$) determined via 5-fold cross-validation on 80 surgical development cases.
- **Production-Grade Microservice**: Lightweight Python HTTP inference microservice with CORS support, sub-15ms warm latency, strict schema validation, and health checks.
- **Zero-Trust Clinical Container**: Minimal Docker image packaged for deployment on **Google Cloud Run** or Kubernetes.

---

## Repository Structure

```
├── Dockerfile                  # Container definition for Google Cloud Run / Docker
├── .dockerignore               # Optimized Docker build context exclusions
├── .gitignore                  # Git exclusions for venv, raw datasets (*.vital), caches
├── README.md                   # Repository documentation
├── docker/
│   └── requirements.txt        # Production inference-only dependencies
├── docs/
│   ├── ISI_DATA_FLOW.md        # Clinical data flow & streaming pipeline architecture
│   └── ISI_SPECIFICATION.md    # Intraoperative Ischemia Severity Index (ISI) mathematical spec
├── models/
│   ├── beatahead_phase5_model.joblib # Calibrated XGBoost frozen model weights
│   ├── feature_schema.json     # 26-feature expected ordering, types, and ranges
│   ├── MODEL_MANIFEST.json     # SHA-256 integrity hashes and provenance
│   └── model_metadata.json     # Hyperparameters, cohort splits, and evaluation metrics
├── reports/                    # Validation reports, audit logs, and benchmark figures
│   ├── figures/                # ROC/PR curves, SHAP explainability plots, PAT sync charts
│   └── phase*.md               # Detailed clinical audit & engineering reports
├── scratch/                    # Audit scripts, cohort replacement analysis, and staging tools
├── src/
│   ├── inference.py            # BeatAheadInferenceEngine: core feature validation & prediction
│   ├── ml_service.py           # Multi-threaded HTTP inference microservice (GET /health, POST /predict)
│   ├── data_download.py        # VitalDB raw dataset acquisition utilities
│   ├── process_100_cohort.py   # Signal preprocessing, beat detection, and window generation
│   └── train_and_evaluate_models.py # Model training, CV tuning, and benchmark scripts
└── tests/
    ├── test_inference.py       # Engine validation & edge-case test suite
    ├── test_container_service.py # HTTP microservice parity, latency, and boundary tests
    ├── test_isi_engine.py      # ISI rule scoring and clinical threshold tests
    ├── test_staging_integration.py # End-to-end integration contracts
    └── test_website_ml_integration.py # Web client & clinician UI communication tests
```

---

## Getting Started

### 1. Installation

Clone the repository and install the runtime dependencies:

```bash
git clone https://github.com/skandakn/ML-Model-Ischemic.git
cd ML-Model-Ischemic

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r docker/requirements.txt
```

### 2. Running Test Suite

Verify local functionality across all 72 automated tests (engine validation, service latency, numerical bit-for-bit parity, and edge-case handling):

```bash
python -m unittest discover tests
```

### 3. Starting the Inference Microservice

Launch the microservice locally:

```bash
python src/ml_service.py --host 127.0.0.1 --port 8000
```

#### Health Check
```bash
curl http://127.0.0.1:8000/health
```

#### Run Prediction
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "ecg_hr_mean": 75.0,
      "ecg_hr_std": 2.5,
      "ecg_rr_sdnn": 35.0,
      "ecg_rr_rmssd": 28.0,
      "ecg_pnn50": 12.0,
      "ecg_r_amp_mv": 1.1,
      "ecg_qrs_width_ms": 88.0,
      "ecg_sqi": 0.95,
      "pat_median_ms": 220.0,
      "pat_iqr_ms": 15.0,
      "pat_valid_fraction": 0.98,
      "pat_valid": 1.0,
      "ppg_pulse_amp": 0.8,
      "ppg_perfusion_index": 2.4,
      "ppg_crest_time_ms": 110.0,
      "ppg_sqi": 0.92,
      "spo2_mean": 98.5,
      "spo2_min": 97.0,
      "spo2_std": 0.5,
      "spo2_desat_count": 0.0,
      "st_obs_mean": -0.4,
      "st_obs_median": -0.4,
      "st_obs_min": -0.6,
      "st_obs_std": 0.1,
      "st_delta_baseline": -0.3,
      "st_slope_mm_min": -0.05
    }
  }'
```

Response:
```json
{
  "alert": 0,
  "probability": 0.0412,
  "threshold": 0.156742,
  "confidence_band": "LOW_RISK",
  "lead_time_seconds": 300,
  "target_horizon_seconds": 300,
  "model_version": "1.0.0",
  "timestamp": "2026-10-13T04:55:00Z"
}
```

---

## Container Deployment

The container is optimized for Google Cloud Run, AWS ECS, or local Docker runtimes:

```bash
# Build Docker image
docker build -t beatahead-ml-service .

# Run container on port 8080
docker run -p 8080:8080 beatahead-ml-service
```

---

## Disclaimer

**Investigational Research Prototype**: BeatAhead is designed for retrospective research, technical benchmarking, and educational purposes. It is **not** cleared by the FDA or CE-marked as a medical device and should not be used for primary diagnostic or autonomous clinical intervention.
