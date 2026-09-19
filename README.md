device: cuda  # or cpu
# PHM-XAI: Engine-Aware Trustworthy Prognostics

An experimental, end-to-end prognostics pipeline for **remaining useful life (RUL)** and **health index (HI)** estimation on the NASA C-MAPSS turbofan dataset. The project extends point predictions with uncertainty estimates, post-hoc explanations, and constraint-aware maintenance recommendations.

The repository is intended for research and engineering evaluation. Reported results are project runs on C-MAPSS, not a claim of production readiness or universal state-of-the-art performance.

## Contents

- [System Overview](#system-overview)
- [System Components](#system-components)
- [Architecture](#architecture)
- [Methodology](#methodology)
- [Implementation Objectives](#implementation-objectives)
- [Quick Setup](#quick-setup)
- [Run the Pipeline](#3-run-the-pipeline)
- [Current Progress](#current-progress)
- [Selected Findings](#selected-findings)
- [Outputs and Repository Hygiene](#outputs-and-repository-hygiene)
- [Tests](#tests)
- [Repository Layout](#repository-layout)

## Implementation positioning

### Implementation approach: engine-aware trustworthy prognostics

The central research idea is to treat prognostics as a **decision-support chain** rather than a point-estimation problem. A maintenance recommendation is considered trustworthy only when four questions are answered together:

1. **What is the predicted degradation state?** Joint RUL and HI estimation provides the point forecast.
2. **How uncertain is that forecast for this operating regime?** Engine-disjoint conformal calibration provides an uncertainty baseline, while the proposed EARA-Conformal extension makes interval size regime- and scale-aware.
3. **Why did the model produce it, and do explanation methods agree?** Integrated Gradients, SHAP, temporal relevance, and ERI expose sensor-level evidence and explanation disagreement.
4. **What action is feasible and who must review it?** A rule-based policy combines health, uncertainty, ERI, and operational constraints into an auditable recommendation.

The implementation uses a system-level reliability layer to improve calibration and decision quality under heterogeneous operating conditions without simply widening every interval or hiding low-confidence explanations. The repository documents implemented components, generated evidence, and remaining validation tasks. See [Implementation map](outputs/reports/research_roadmap.md) and the objective reports for scope and usage.

## System Overview

```mermaid
flowchart LR
    A[NASA C-MAPSS raw data] --> B[Validation and preprocessing]
    B --> C[Windowed sensor sequences]
    C --> D[Multi-task prognostics model]
    D --> E[RUL and HI predictions]
    E --> F[EARA-Conformal uncertainty]
    E --> G[IG, SHAP, temporal relevance, ERI]
    F --> H[Decision state]
    G --> H
    H --> I[Rules and operational constraints]
    I --> J[Auditable recommendation]
```

The main processing stages are:

1. Load and validate FD001, FD002, FD003, or FD004.
2. Generate capped RUL and normalized HI labels, select features, scale data, and create sliding windows.
3. Train a model with shared representations and RUL/HI prediction heads.
4. Evaluate point predictions with MAE, RMSE, R², and NASA Score.
5. Calibrate engine-disjoint conformal intervals for RUL.
6. Produce sensor and temporal explanations for RUL and HI.
7. Fuse RUL, HI, interval width, ERI, and operational constraints into a maintenance recommendation.

## System Components

The proposed workflow is a layered decision-support system rather than a fully autonomous maintenance controller:

- **Prognostics:** LSTM, GRU, Transformer, TCN+GRU Hybrid, and GRU-Att-Deg models support joint RUL and HI estimation.
- **Uncertainty:** The implemented baseline is split conformal prediction; the EARA-Conformal module adds heteroscedastic scaling and k-means operating-regime calibration. Engine-disjoint calibration is retained, but full multi-seed validation of EARA is still pending.
- **Explainability:** Integrated Gradients and sensor-level Kernel SHAP are combined with temporal relevance. ERI summarizes agreement between explanation methods and is used as a review signal, not as proof of causal correctness.
- **Decision support:** A rule-based engine classifies health and uncertainty, applies operational constraints, and flags ambiguous cases for human review. The HITL update path remains a research stub and is not claimed as online learning.

The decision engine is deliberately auditable: each result includes inputs, health state, uncertainty level, risk components, reasons, constraint violations, and the recommended action.

## Architecture

```text
data/raw/                  NASA C-MAPSS text files (not committed)
    |
src/preprocessing/         validation, labels, feature selection, scaling, windows
    |
data/processed/            NumPy windows, labels, engine IDs, metadata (local)
    |
src/models/                LSTM, GRU, Transformer, Hybrid, GRU-Att-Deg
    |
outputs/checkpoints/       best model checkpoints (local)
    |
  +-----+----------------------+------------------+
  |                            |                  |
src/uncertainty/       src/explainability/  src/decision_engine/
  |                            |                  |
RUL intervals          attributions + ERI   recommendations + constraints
  +----------------------------+------------------+
                   |
              outputs/results, reports,
              figures, xai, and logs
```

## Methodology

### Data and Labels

NASA C-MAPSS contains multivariate engine run-to-failure trajectories under four operating and fault-condition subsets. The preprocessing pipeline validates frames, removes near-constant features, scales retained features, and creates fixed-length windows. The current training pipeline uses a 40-cycle window; some reports contain earlier 30-cycle runs, so reproduce results with the matching generated data and configuration.

Training labels use a piecewise-linear RUL target capped at 125 cycles. HI is normalized to `[0, 1]` and learned as an auxiliary target.

### Multi-Task Models

The models share a sequence encoder and predict RUL and HI together. The GRU-Att-Deg variant adds temporal attention, a degradation-aware HI head, and a softplus scale head used by the uncertainty pipeline. Training uses grouped engine-level validation so windows from one engine do not cross the train/validation boundary.

### Uncertainty and Explanations

EARA-Conformal fits regimes on training-engine features, calibrates normalized residuals on held-out calibration engines, and produces adaptive RUL intervals. The implementation evaluates empirical coverage, coverage error, MPIW, PINAW, and per-regime coverage.

The XAI pipeline generates Integrated Gradients, Kernel SHAP, temporal relevance, plots, `.npz` attribution arrays, ERI JSON, and HTML reports for RUL and HI targets. Detailed limitations are documented in [Objective 3](outputs/reports/objective_3_report.md).

## Implementation Objectives

| Objective | Implementation focus | Current status |
| --- | --- | --- |
| [O1: multi-task prognostics](outputs/reports/objective1_report.md) | Can a shared sequence model estimate RUL and HI across all C-MAPSS regimes? | Implemented and benchmarked; hybrid underperformance is documented. |
| [O2: calibrated uncertainty](outputs/reports/objective_2_report.md) | How does the implementation calibrate uncertainty at engine and operating-regime levels? | Baseline implemented; adaptive and engine-block validation remains. |
| [O3: explanation reliability](outputs/reports/objective_3_report.md) | How does the implementation expose sensor evidence and review signals? | Pipeline implemented; faithfulness and stability validation remains. |
| [O4: decision intelligence](outputs/reports/objective_4_report.md) | How does the implementation produce auditable actions from model outputs? | Rule engine implemented and tested; HITL learning is not implemented. |

## Quick Setup

### 1. Install

Python 3.9+ is recommended. On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` currently pins a CUDA 12.8 PyTorch build. Use a compatible PyTorch build for a CPU-only or different CUDA environment, then install the remaining requirements.

### 2. Add the dataset

The raw NASA C-MAPSS files are not included. Place all four subsets in `data/raw/`:

```text
data/raw/
├── train_FD001.txt       test_FD001.txt       RUL_FD001.txt
├── train_FD002.txt       test_FD002.txt       RUL_FD002.txt
├── train_FD003.txt       test_FD003.txt       RUL_FD003.txt
└── train_FD004.txt       test_FD004.txt       RUL_FD004.txt
```

### 3. Run the Pipeline

```powershell
# Exploratory data analysis for FD001
python main.py

# Preprocess one subset, or use --subset all
python scripts/preprocess.py --subset FD001

# Train and evaluate a model
python scripts/train.py --model gru --subset FD001
python scripts/evaluate.py --model gru --subset FD001

# RUL uncertainty intervals
python scripts/uncertainty.py --model gru --subset FD001 --n_regimes 7

# XAI for one target; use --subset all --model all --target all for a full run
python scripts/explain.py --model gru_att_deg --subset FD001 --target rul
```

Most stages require processed arrays and/or a trained checkpoint from the previous stage. Model and subset names are `lstm`, `gru`, `transformer`, `hybrid`, `gru_att_deg` and `FD001`–`FD004`.

### 4. Run decision support

The decision engine accepts one record, a list of records, or an object containing a `records` list. Example inputs are provided in `data/external/`.

```powershell
python scripts/decision.py --subset all

python scripts/decision.py `
  --subset FD001 `
  --minimum-safe-rul 10 `
  --maintenance-lead-time 10 `
  --no-maintenance-window
```

Each record should contain `rul`, `hi`, `rul_lower`, `rul_upper`, and `eri`; `top_k_sensors` and engine identifiers are optional. Results are written under `outputs/results/` and human-readable reports under `outputs/reports/`.

### 5. Run the HITL review workflow

The HITL runner consumes the JSON response generated by `scripts/decision.py`,
creates pending review items for decisions flagged by the O4 engine, and
writes JSON results, an HTML report, and a log file:

```powershell
python scripts/hitl.py --subset FD004 --model gru
python scripts/hitl.py --subset all --model all

# Resolve a review and regenerate the report
python scripts/hitl.py `
  --input outputs/results/decision_FD004_gru.json `
  --review-id decision-0-0 `
  --reviewer operator-1 `
  --action APPROVE `
  --comment "Reviewed by maintenance control"
```

Supported subsets are `FD001` through `FD004`; supported models are `lstm`,
`gru`, `transformer`, `hybrid`, and `gru_att_deg`. Missing artifacts are
reported as skips when using `all`. Use `APPROVE`, `OVERRIDE` with
`--override-action`, or `REJECT` to resolve a queued review. Resolutions are
stored in `outputs/results/hitl_feedback/` and reflected in the HTML report.
HITL decisions do not update model weights automatically.

## Current Progress

The following status reflects the project reports and implemented modules:

| Area | Status | Current evidence |
| --- | --- | --- |
| Preprocessing and window generation | Implemented | Validation, labeling, scaling, feature selection, and engine metadata are present. |
| Joint RUL/HI prediction | Implemented and evaluated | Five model families evaluated on FD001–FD004 in the project reports. |
| Uncertainty quantification | Implemented and evaluated | Split, adaptive, CQR, and EARA-Conformal components are present; the current uncertainty script defaults to 7 regimes. |
| Explainability | Implemented and evaluated | IG, SHAP, temporal relevance, ERI, plots, and HTML reporting are present. |
| Decision support | Implemented and tested | Rules, constraints, structured JSON results, HTML reports, and decision tests are present. |
| HITL updating | Scaffold only | Feedback storage and model update functions are placeholders; no online adaptation result is claimed. |
| Extended validation | In progress | Multi-seed stability, ablations, faithfulness checks, and external-dataset validation remain open items. |

## Selected findings

These are compact excerpts from the local project reports, included to make the current state visible without presenting every experiment:

- **Point prediction:** On the reported FD001 run, GRU-Att-Deg achieved RUL MAE `10.050`, RUL RMSE `13.644`, R² `0.884`, and NASA Score `308.63`. On FD004, the reported Transformer run was strongest among the evaluated models with RUL MAE `13.559` and R² `0.808`.
- **Uncertainty:** The canonical GRU split-conformal baseline reaches 90% empirical coverage of `0.880`, `0.897`, `0.850`, and `0.928` on FD001–FD004, with MPIW of `42.06`, `57.88`, `37.87`, and `70.81` cycles. Adaptive EARA figures should not be reported as validated until the corresponding run manifest and repeated engine-level evaluation are archived.
- **Explainability:** The reported mean ERI was `0.880` for HI and `0.872` for RUL across the four subsets and five models. FD003 showed weaker explanation agreement than the other subsets, so its explanations should be reviewed more cautiously.
- **Decision support:** The engine produces explicit actions such as `CONTINUE_OPERATION`, `INSPECT`, `SCHEDULE_MAINTENANCE`, and `URGENT_MAINTENANCE`, with constraint overrides and human-review flags instead of hiding those decisions in a single score.

The numbers above depend on the data preprocessing, checkpoint, seed, and report run. Re-run the relevant scripts before using them as a formal comparison.

## Outputs and repository hygiene

Generated artifacts are intentionally local. The ignore rules exclude `outputs/`, raw and processed data, checkpoints, logs, `docs/`, virtual environments, and archives from version control. Typical generated locations are:

```text
outputs/
├── checkpoints/   trained model weights
├── figures/       EDA and experiment plots
├── logs/          pipeline logs
├── models/        scalers and related artifacts
├── reports/       CSV and HTML reports
├── results/       metrics, predictions, intervals, decisions
└── xai/           attribution arrays, plots, ERI, HTML reports
```

The tracked source of truth is the code, configuration, tests, and this README. Local reports under `docs/` are useful for reproducing project context but are not distributed by Git according to the current `.gitignore`.

## Tests

Run the test suite from the repository root:

```powershell
python -m pytest -q
```

Some integration paths require generated processed data or checkpoints. A clean checkout therefore needs the dataset and the corresponding pipeline stages before those paths can run.

## Repository layout

```text
configs/       subset configuration files
data/          external inputs, raw data, and processed arrays
scripts/       EDA, preprocessing, training, evaluation, XAI, UQ, and decisions
src/           reusable preprocessing, models, uncertainty, XAI, HITL, and decision code
tests/         preprocessing, model, uncertainty, XAI, and decision tests
docs/          local architecture, methodology, and objective reports
outputs/       generated artifacts; ignored by Git
```

See `requirements.txt` for the complete dependency list. For a concise implementation map, read [Implementation map](outputs/reports/research_roadmap.md).
