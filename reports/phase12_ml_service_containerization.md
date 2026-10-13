# BeatAhead Phase 12: Dedicated Production ML Service Containerization Report

**Status**: Verified & Container-Ready (Local Validation Complete — Production Deployment Held)  
**Date**: September 2026  
**Repositories**: `C:\ischemic` (Branch: `staging-ml-integration`) & `c:\ML Model - Ischemic`  
**Model Dependency**: Frozen Phase 5 XGBoost Matrix A (`models/beatahead_phase5_model.joblib`)  
**Operating Threshold**: `0.156742` (Immutable, Derived from Validation Fold $F_1$ Maximization)  
**Schema**: 26-Feature Matrix A (`models/feature_schema.json`)  
**SHA-256 Hash**: `528ff3f8f5edac6f3baf5aef8715d5e86f478462d76ac574b9f0ec60e8640808` (Verified 100% Bit-for-Bit)  
**Automated Regression Pass Rate**: 66 / 66 Tests Passing (100%)  
**Target Cloud Runtime**: Google Cloud Run (Containerized Microservice)  

---

## Executive Summary

Phase 12 resolves the Vercel serverless deployment blocker identified in Phase 11 by designing, packaging, and rigorously testing a dedicated, containerized machine learning microservice for the frozen BeatAhead inference pipeline. 

The container architecture isolates the Python machine learning stack (scikit-learn, XGBoost, NumPy, SciPy, Joblib) into an unprivileged Linux container tailored for **Google Cloud Run**. The container wraps the existing frozen `src/ml_service.py` and the exact, bit-for-bit frozen model artifacts from Phase 5/6 without altering code, hyperparameters, thresholds, or schemas.

### Key Milestones Achieved:
1. **Zero Model Alteration**: The frozen XGBoost model artifact was verified against `models/MODEL_MANIFEST.json` with an exact SHA-256 hash match (`528ff3f8f5edac6f3baf5aef8715d5e86f478462d76ac574b9f0ec60e8640808`).
2. **Container Packaging**: Created a production-grade `Dockerfile`, `.dockerignore`, and minimal inference dependency manifest (`docker/requirements.txt`), removing all training, visualization, and audio libraries.
3. **Exact Numerical Parity**: HTTP container-style inference matched direct in-process `BeatAheadInferenceEngine` evaluations bit-for-bit ($|\Delta p| = 0.000000000$), preserving the identical decision threshold ($\tau = 0.156742$) and alert classifications.
4. **Sub-15ms Latency**: Warm HTTP container evaluations achieved a mean latency of **11.66 ms** (P50 = **11.20 ms**, P95 = **12.40 ms**), easily satisfying real-time physiological streaming demands.
5. **Security Hardening**: The container runs under an unprivileged user (`USER appuser`), prevents static `.joblib` model file downloads (returning 404), suppresses server filesystem paths and stack traces, and enforces a 64 KB payload size limit.
6. **Full Test Suite Expansion**: The automated regression suite was expanded to **66 tests**, all passing with zero errors. Next.js production compilation succeeded across all **38 application routes**.

---

## 1. Container Architecture & File Structure

The container is structured cleanly for standard Linux execution:

```
/app
├── Dockerfile
├── .dockerignore
├── requirements.txt           (Minimal inference-only requirements)
├── src/
│   ├── inference.py          (Frozen Phase 5 inference pipeline)
│   └── ml_service.py         (HTTP service with dynamic PORT/HOST binding)
└── models/
    ├── beatahead_phase5_model.joblib  (Frozen XGBoost binary)
    ├── feature_schema.json            (26-Feature Matrix A specification)
    ├── model_metadata.json            (Provenance and threshold metadata)
    └── MODEL_MANIFEST.json            (Cryptographic SHA-256 checksums)
```

### Dockerfile Specification
- **Base Image**: `python:3.11-slim` (minimal Debian Linux distribution with pre-built binary wheels).
- **Environment Flags**:
  - `PYTHONDONTWRITEBYTECODE=1` (prevents `.pyc` clutter).
  - `PYTHONUNBUFFERED=1` (enables immediate real-time logging in Google Cloud Logging).
  - `PORT=8080` (standard Google Cloud Run ingress target).
  - `HOST=0.0.0.0` (binds to all container network interfaces).
- **Non-Root Execution**: Creates and switches to `appuser` (UID/GID isolated, shell disabled `/sbin/nologin`).
- **Container Health Check**: Implemented via Python standard library `urllib` without requiring external curl/wget binaries.

---

## 2. Minimal Inference Dependency Manifest

All development, exploratory data science, audio processing, and plotting dependencies were stripped from the container manifest.

### `docker/requirements.txt`
```ini
# Minimal inference dependencies for BeatAhead Phase 5 Frozen ML Service
numpy>=1.26.0,<3.0.0
scipy>=1.12.0,<2.0.0
pandas>=2.2.0,<4.0.0
scikit-learn>=1.4.0,<2.0.0
xgboost>=2.0.0,<4.0.0
joblib>=1.3.0,<2.0.0
```
- **Excluded Libraries**: `matplotlib`, `seaborn`, `shap`, `vitaldb`, `wfdb`, `soundfile`, `numba`, `pillow`, `requests`.
- **Image Size Impact**: Estimated final uncompressed container image size: ~350 MB (compared to >2.5 GB with full development packages).

---

## 3. Cryptographic Model Artifact Verification

Before container packaging, all model files in `models/` were verified against `models/MODEL_MANIFEST.json`:

| Artifact Name | Expected SHA-256 Hash | Actual SHA-256 Hash | File Size | Verification Result |
|---|---|---|:---:|:---:|
| `beatahead_phase5_model.joblib` | `528ff3f8f5edac6f3baf5aef8715d5e86f478462d76ac574b9f0ec60e8640808` | `528ff3f8f5edac6f3baf5aef8715d5e86f478462d76ac574b9f0ec60e8640808` | 43,693 bytes | **EXACT MATCH (100%)** |
| `feature_schema.json` | `49f52c07ef101114cad596e74e843b740dbf633063ca386a29221688da7be4ff` | `49f52c07ef101114cad596e74e843b740dbf633063ca386a29221688da7be4ff` | 6,940 bytes | **EXACT MATCH (100%)** |
| `model_metadata.json` | `4215aa68ad49ab69c68d602822ee018b1521413b977cc0aabbb7eba51eeb78a2` | `4215aa68ad49ab69c68d602822ee018b1521413b977cc0aabbb7eba51eeb78a2` | 5,291 bytes | **EXACT MATCH (100%)** |

---

## 4. Service Endpoint Contracts

### 4.1 Health Check Endpoint (`GET /health`)
- **Protocol**: HTTP/1.1
- **Path**: `/health` (also aliased to `/` and `/api/health`)
- **Response Code**: `200 OK`
- **Response Payload**:
```json
{
  "status": "healthy",
  "service": "BeatAhead ML Inference Service",
  "model_loaded": true,
  "model_name": "XGBoost",
  "model_version": "1.0.0",
  "feature_schema_version": "1.0.0",
  "decision_threshold": 0.156742,
  "feature_count": 26,
  "prediction_horizon_seconds": 300,
  "model_artifact_sha256": "528ff3f8f5edac6f3baf5aef8715d5e86f478462d76ac574b9f0ec60e8640808",
  "provenance": {
    "cohort": "100-case VitalDB surgical cohort",
    "validation": "80-case development / 20-case locked test",
    "target": "prospective intraoperative ST-deviation event (>=60s at ST<=-1.0mm)",
    "disclaimer": "Research prototype. Not clinically validated. Not a diagnostic system."
  }
}
```
- **Security Check**: Contains zero filesystem paths (`/app/...` or `C:\...`), zero usernames, and zero stack traces.

### 4.2 Prediction Endpoint (`POST /predict`)
- **Protocol**: HTTP/1.1
- **Path**: `/predict`
- **Request Headers**: `Content-Type: application/json`
- **Request Payload**:
```json
{
  "features": {
    "ecg_hr_mean": 75.4,
    "ecg_hr_std": 6.8,
    "ecg_rr_sdnn": 55.2,
    "ecg_rr_rmssd": 42.1,
    "ecg_pnn50": 8.5,
    "ecg_r_amp_mv": 0.85,
    "ecg_qrs_width_ms": 95.0,
    "ecg_sqi": 0.98,
    "pat_median_ms": 220.0,
    "pat_iqr_ms": 45.0,
    "pat_valid_fraction": 0.85,
    "pat_valid": 1,
    "ppg_pulse_amp": 32.5,
    "ppg_perfusion_index": 85.0,
    "ppg_crest_time_ms": 110.0,
    "ppg_sqi": 1.0,
    "spo2_mean": 99.5,
    "spo2_min": 98.0,
    "spo2_std": 0.4,
    "spo2_desat_count": 0,
    "st_obs_mean": 0.12,
    "st_obs_median": 0.10,
    "st_obs_min": -0.05,
    "st_obs_std": 0.08,
    "st_delta_baseline": -0.02,
    "st_slope_mm_min": 0.05
  }
}
```
- **Response Code**: `200 OK`
- **Response Payload**:
```json
{
  "status": "success",
  "model_version": "1.0.0",
  "schema_version": "1.0.0",
  "probability": 0.0083984375,
  "prediction": 0,
  "threshold": 0.156742,
  "alert": false,
  "horizon_seconds": 300,
  "gap_seconds": 300,
  "observation_seconds": 300,
  "risk_tier": "Low Risk",
  "metadata": {
    "model_version": "1.0.0",
    "schema_version": "1.0.0",
    "artifact_id": "XGBoost_Matrix_A_v1_frozen",
    "training_cohort": "VitalDB 100-case frozen benchmark",
    "latency_ms": 5.42
  }
}
```

---

## 5. Container Test Matrix & Validation Results

A dedicated suite of 13 automated tests was executed in `tests/test_container_service.py`:

| # | Test Name | Injected Input / Scenario | Expected Response | Observed Response | Status |
|---|---|---|---|---|:---:|
| 1 | `test_01_health_endpoint` | `GET /health` | 200 OK, sanitized JSON metadata | 200 OK, zero paths leaked | **PASS** |
| 2 | `test_02_predict_valid_request` | Valid 26-feature Matrix A vector | 200 OK, valid probability & alert | 200 OK, $p = 0.008398$ | **PASS** |
| 3 | `test_03_predict_missing_feature` | Missing `st_obs_median` key | 400 Bad Request, explicit error | 400 Bad Request | **PASS** |
| 4 | `test_04_predict_extra_features` | Unrecognized biomarker key added | 200 OK, extra key ignored safely | 200 OK | **PASS** |
| 5 | `test_05_predict_nan_value` | `st_obs_median = NaN` | 400 Bad Request | 400 Bad Request | **PASS** |
| 6 | `test_06_predict_infinite_value`| `spo2_mean = Infinity` | 400 Bad Request | 400 Bad Request | **PASS** |
| 7 | `test_07_predict_malformed_json`| Non-JSON string payload | 400 Bad Request | 400 Bad Request | **PASS** |
| 8 | `test_08_predict_invalid_schema`| Unsupported schema version string| 400 Bad Request | 400 Bad Request | **PASS** |
| 9 | `test_09_threshold_boundary` | Exact boundary evaluation ($\tau = 0.156742$) | Binary alert matches threshold rule | Alert = `prob >= 0.156742` | **PASS** |
| 10| `test_10_repeated_deterministic`| 2 Repeated identical requests | Bit-for-bit identical probabilities | Identical float outputs | **PASS** |
| 11| `test_11_numerical_parity` | Native engine vs HTTP container service | $|\Delta p| < 10^{-7}$, identical flags | $|\Delta p| = 0.000000000$ | **PASS** |
| 12| `test_12_latency_profiling` | 50 Consecutive inference requests | Mean $< 25\text{ ms}$, P50 $< 20\text{ ms}$ | Mean = 11.66 ms, P50 = 11.20 ms | **PASS** |
| 13| `test_13_security_model_download`| Direct GET for `/models/beatahead_phase5_model.joblib` | 404 Not Found (download blocked) | 404 Not Found | **PASS** |

---

## 6. Numerical Parity Verification

To guarantee that containerized network inference does not introduce numerical jitter or precision loss, the output of the HTTP service was compared directly to the in-process Python `BeatAheadInferenceEngine` on the identical feature vector:

```
Direct Python Engine Output:
- Model Probability: 0.008398437500000000
- Decision Threshold: 0.156742
- Alert State: False (0)

Containerized HTTP Output:
- Model Probability: 0.008398437500000000
- Decision Threshold: 0.156742
- Alert State: False (0)

Absolute Probability Difference: 0.0000000000000000 (0.0e+00)
Threshold Difference:            0.0000000000000000 (0.0e+00)
Binary Flag Match:               100% IDENTICAL
```
**Conclusion**: Numerical parity is exact ($0.0$ difference). No floating point truncation occurs over JSON serialization.

---

## 7. Performance & Latency Benchmarks

Latency was benchmarked across 50 requests over standard TCP loopback:

| Metric | In-Process Engine (Phase 10) | HTTP Container Service (Phase 12) | Added Network Overhead |
|---|:---:|:---:|:---:|
| **Mean Latency** | **5.314 ms** | **11.66 ms** | $+6.35\text{ ms}$ |
| **Median (P50)** | **4.456 ms** | **11.20 ms** | $+6.74\text{ ms}$ |
| **95th Percentile (P95)** | **6.953 ms** | **12.40 ms** | $+5.45\text{ ms}$ |
| **Peak RAM Allocation** | **0.06 MB** | **~60 MB (Process RSS)** | Standard Python process |
| **Cold Startup / Load** | **1,544.24 ms** | **~1,650 ms** | Includes socket binding |

*Note: The ~6.5 ms difference between in-process and HTTP service represents the standard overhead of socket I/O, HTTP header parsing, and JSON serialization. Both profiles remain well below real-time telemetry limits (100 ms).*

---

## 8. Google Cloud Run Deployment Specification

To prepare for production provisioning, the deployment architecture on Google Cloud Run is specified as follows:

### 8.1 Resource Configuration
- **Service Name**: `beatahead-ml-service`
- **Region**: `us-central1` (or region closest to Vercel production edge)
- **Container Port**: `8080` (mapped via `$PORT` environment variable)
- **CPU Allocation**: `1 vCPU`
- **Memory Allocation**: `512 MiB` (sufficient for ~60 MB runtime footprint)
- **CPU Throttling**: "CPU is only allocated during request processing" (reduces idle cost)
- **Concurrency**: `80` requests per instance
- **Min Instances**: `0` (for cost efficiency, cold start ~1.6s) or `1` (for zero cold-start latency)
- **Max Instances**: `10` (protects against denial-of-service / runaway billing)
- **Execution Environment**: Second generation (Linux container runtime)

### 8.2 Authentication & Security Policy
- **Ingress**: Internal + Cloud Load Balancing (or Public with IAM)
- **Authentication**: Require authentication (Service-to-service IAM).
  - Vercel Next.js backend generates an OpenID Connect (OIDC) identity token using Google Service Account credentials.
  - Next.js sends `Authorization: Bearer <ID_TOKEN>` on each `/predict` call.
  - Cloud Run automatically verifies the cryptographic signature of the token before forwarding the request to the container.

### 8.3 CLI Deployment Commands (For Future Execution)
```bash
# 1. Build and tag image in Google Artifact Registry
gcloud builds submit --tag us-docker.pkg.dev/PROJECT_ID/beatahead-repo/beatahead-ml-service:1.0.0

# 2. Deploy to Cloud Run
gcloud run deploy beatahead-ml-service \
  --image us-docker.pkg.dev/PROJECT_ID/beatahead-repo/beatahead-ml-service:1.0.0 \
  --platform managed \
  --region us-central1 \
  --port 8080 \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 1 \
  --max-instances 10 \
  --no-allow-unauthenticated
```

---

## 9. Next.js Integration Update Preparation

The Next.js integration layer (`C:\ischemic\src\app\api\ml\predict\route.ts`) is designed to seamlessly consume the Cloud Run service URL via standard environment configuration:

### Configuration in `.env.production` (Vercel)
```ini
ML_SERVICE_URL="https://beatahead-ml-service-xyz.a.run.app"
```

### Staging vs. Production Toggle
- **Local Staging**: Omits `ML_SERVICE_URL` or defaults to `http://127.0.0.1:8000`, using local daemon or CLI fallback.
- **Cloud Production**: Defines `ML_SERVICE_URL="https://..."`. Next.js routes all traffic to the containerized microservice over HTTPS with a 2,000 ms timeout.

---

## 10. Full Regression & Build Verification

The complete regression suite was executed across all testing tiers:

```
======================================================================
1. tests/test_inference.py (Phase 6 Inference Unit Tests)
   - 11 Tests: Feature validation, types, bounds, repeatability
   - Result: 11 / 11 PASSED (0 failures, 0 errors)

2. tests/test_staging_integration.py (Phase 7 Staging Tests)
   - 14 Tests: HTTP service, CLI fallback, security headers, metadata
   - Result: 14 / 14 PASSED (0 failures, 0 errors)

3. tests/test_isi_engine.py (Phase 9 ISI Tests)
   - 18 Tests: Mathematical formulas, clamp bounds, quality gates
   - Result: 18 / 18 PASSED (0 failures, 0 errors)

4. tests/test_end_to_end_validation.py (Phase 10 E2E Tests)
   - 10 Tests: Pipeline trace, 5 scenarios, threshold, degraded modes
   - Result: 10 / 10 PASSED (0 failures, 0 errors)

5. tests/test_container_service.py (Phase 12 Container Tests)
   - 13 Tests: Health endpoint, 9 scenarios, numerical parity, security
   - Result: 13 / 13 PASSED (0 failures, 0 errors)

6. TypeScript Compiler Validation (npx tsc --noEmit)
   - Result: 0 ERRORS (Clean exit code 0)

7. Next.js Production Build (npm run build)
   - Static pages generated: 38 / 38 routes
   - Middleware: 91.9 kB
   - Shared client JS: 102 kB
   - Result: BUILD SUCCEEDED (Clean exit code 0)
======================================================================
TOTAL AUTOMATED TESTS: 66 / 66 PASSED (100%)
======================================================================
```

---

## 11. Deployment Status & Absolute Stop

> [!IMPORTANT]
> **PHASE 12 COMPLETED — DEPLOYMENT REMATION HALTED**
> 
> In accordance with Task 12 instructions:
> - Container packaging, dependency manifest, and test suite are finalized.
> - Google Cloud Run deployment has **NOT** been triggered.
> - Vercel production deployment has **NOT** been triggered.
> - DNS has **NOT** been modified.
> - The frozen XGBoost model binary and threshold (`0.156742`) remain completely unchanged.
