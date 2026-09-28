<<<<<<< HEAD
# PHM-XAI — Prognostics and Health Monitoring for Aircraft Engines

A research framework for **predictive maintenance** on multivariate sensor time-series from the **NASA C-MAPSS** turbofan engine degradation dataset. PHM-XAI delivers a complete **multi-task deep learning pipeline** for joint **Remaining Useful Life (RUL)** and **Health Index (HI)** prediction using LSTM, GRU, Transformer, and Hybrid architectures.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)

---

## Table of Contents

- [What is PHM-XAI?](#what-is-phm-xai)
- [How It Works](#how-it-works)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Testing](#testing)
- [Future Development](#future-development)
- [Contributing](#contributing)
- [Citation](#citation)

---

## What is PHM-XAI?

PHM-XAI is a modular **prognostics and health monitoring (PHM)** framework built for aircraft engine degradation modeling. It ingests NASA C-MAPSS sensor readings and produces two complementary outputs:

1. **Remaining Useful Life (RUL)** — estimated cycles until engine failure (regression)
2. **Health Index (HI)** — normalized engine health score from 0.0 to 1.0 (regression)

### Real-World Applications

- **Aerospace maintenance** — schedule inspections before critical failure
- **Fleet monitoring** — track degradation trends across multiple engines
- **Research benchmarking** — compare recurrent and attention-based prognostics models
- **Predictive maintenance pipelines** — foundation layer for downstream decision systems

### Current Scope (Objective 1 — Complete)

The implemented pipeline covers the full **data-to-evaluation** workflow:

| Stage | Status |
|-------|--------|
| Data loading and validation | Complete |
| RUL and HI label generation | Complete |
| Feature selection and scaling | Complete |
| Sliding-window sequence generation | Complete |
| Multi-task model training (LSTM, GRU, Transformer, Hybrid) | Complete |
| Model evaluation with NASA scoring | Complete |

---

## How It Works

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    NASA C-MAPSS Raw Data                    │
│         train_FD00X.txt │ test_FD00X.txt │ RUL_FD00X.txt    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   PREPROCESSING PIPELINE                    │
│  Load → Validate → Label (RUL/HI) → Feature Selection       │
│       → StandardScaler → Sliding Windows → Save .npy        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   MULTI-TASK PROGNOSTICS                    │
│  Engine-wise Train/Val Split → Model Selection              │
│  LSTM │ GRU │ Transformer │ Hybrid                          │
│  MultiTaskLoss (α·RUL + β·HI) → Early Stopping → Checkpoint │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      EVALUATION                             │
│  Test Inference → MAE, RMSE, R², NASA Score → Save Results  │
└─────────────────────────────────────────────────────────────┘
```

### Step-by-Step Process

1. **Load** — Read C-MAPSS train, test, and RUL files into structured DataFrames
2. **Validate** — Check schema, missing values, and column integrity
3. **Label** — Compute RUL (capped at 125 cycles) and normalized HI per engine
4. **Preprocess** — Remove low-variance sensors, apply StandardScaler (fit on train only)
5. **Window** — Generate 30-step sliding windows for training; final window per engine for test
6. **Train** — Multi-task learning with engine-level train/validation split (no data leakage)
7. **Evaluate** — Run inference on test set and compute RUL/HI metrics including NASA score
8. **Save** — Store checkpoints, predictions, training history, and logs under `outputs/`

---

## Key Features

### Data Processing

- **NASA C-MAPSS support** — FD001, FD002, FD003, FD004 subsets
- **Automated validation** — schema checks, missing value detection, summary reports
- **RUL labeling** — piecewise linear RUL with 125-cycle cap (standard C-MAPSS convention)
- **Health Index** — per-engine normalized degradation score
- **Variance-based feature selection** — removes uninformative sensors automatically
- **Sliding-window generation** — configurable window size and stride with engine ID tracking
- **EDA utilities** — lifetime plots, sensor trends, correlation heatmaps, variance analysis

### Multi-Task Deep Learning

- **Four model architectures** — LSTM, GRU, Transformer, and Hybrid (LSTM + Transformer fusion)
- **Shared encoder design** — single representation with separate RUL and HI prediction heads
- **Multi-task loss** — weighted combination of RUL and HI MSE losses
- **Engine-wise splitting** — GroupShuffleSplit ensures no engine appears in both train and validation
- **Training controls** — early stopping, learning rate scheduling, best-checkpoint saving
- **Reproducibility** — fixed random seeds across Python, NumPy, and PyTorch

### Evaluation

- **RUL metrics** — MAE, RMSE, R², NASA C-MAPSS asymmetric scoring function
- **HI metrics** — MAE, RMSE, R²
- **Artifact export** — predictions, training history, and evaluation logs saved automatically

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.9+** | Core language |
| **PyTorch 1.13+** | Deep learning models and training |
| **NumPy / Pandas** | Data manipulation and array storage |
| **scikit-learn** | Scaling, feature selection, train/val splitting |
| **SciPy** | Scientific computing utilities |
| **Matplotlib / Seaborn** | Exploratory data visualization |
| **PyYAML** | Experiment configuration |
| **pytest** | Automated testing |
| **joblib** | Scaler persistence |

---

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ENTRY POINTS                           │
│  main.py │ preprocess.py │ train.py │ evaluate.py           │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                    CORE LIBRARY (src/)                      │
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │  preprocessing/ │  │     models/     │                   │
│  │  Loader         │  │  LSTM / GRU     │                   │
│  │  Validator      │  │  Transformer    │                   │
│  │  LabelGenerator │  │  Hybrid         │                   │
│  │  FeatureScaler  │  │  MultiTaskLoss  │                   │
│  │  WindowGenerator│  │  ModelTrainer   │                   │
│  └─────────────────┘  │  Evaluator      │                   │
│                       └─────────────────┘                   │
│  ┌─────────────────┐                                        │
│  │     utils/      │  config, logging, seed, I/O             │
│  └─────────────────┘                                        │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                      DATA & OUTPUTS                         │
│  data/raw/ │ data/processed/ │ outputs/checkpoints/         │
│  outputs/results/ │ outputs/logs/ │ outputs/figures/        │
└─────────────────────────────────────────────────────────────┘
```

### Model Architecture (Multi-Task Design)

All four models follow the same interface: `forward(x) → (pred_rul, pred_hi)`.

```
Input Tensor (batch, 30, features)
              │
              ▼
     ┌─────────────────┐
     │  Shared Encoder  │  ← LSTM / GRU / Transformer / Hybrid
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │  Shared FC Layer │  Linear → BatchNorm → ReLU → Dropout
     └────────┬────────┘
              │
       ┌──────┴──────┐
       ▼             ▼
   RUL Head       HI Head
  (regression)  (regression)
```

**Hybrid model** runs parallel LSTM and Transformer branches, concatenates their representations, and fuses them before the task heads.

### Preprocessing Pipeline

```
CMAPSSLoader
     ↓
DatasetValidator → summary reports
     ↓
LabelGenerator (RUL + HI)
     ↓
FeatureSelector (variance threshold)
     ↓
FeatureScaler (StandardScaler, train-only fit)
     ↓
WindowGenerator
     ├── Train: sliding windows (stride=1)
     └── Test:  final window per engine
     ↓
Save .npy arrays + metadata.json
```

---

## Project Structure

```
phm_project/
├── main.py                     # EDA entry point
├── configs/                    # YAML experiment configs (FD001–FD004)
├── data/
│   ├── raw/                    # NASA C-MAPSS source files (not in repo)
│   └── processed/              # Generated .npy arrays
├── scripts/
│   ├── preprocess.py           # Full preprocessing pipeline
│   ├── train.py                # Model training
│   └── evaluate.py             # Model evaluation
├── src/
│   ├── preprocessing/          # Data loading, labeling, windowing
│   ├── models/                 # LSTM, GRU, Transformer, Hybrid
│   ├── evaluation/             # Metrics and benchmarking utilities
│   └── utils/                  # Config, logging, seed, I/O
├── tests/                      # pytest test suite
├── outputs/                    # Checkpoints, results, logs, figures
├── requirements.txt
├── CONTRIBUTING.md
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- pip
- CUDA-compatible GPU (recommended for training)

### Installation

```bash
git clone <repository-url>
cd phm_project

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Verify PyTorch and GPU availability:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### Dataset Setup

Download the [NASA C-MAPSS Turbofan Engine Degradation Dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) and place files in:

```
data/raw/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
└── ... (repeat for FD002, FD003, FD004)
```

### Run the Pipeline

```bash
# 1. Preprocess a dataset subset
python scripts/preprocess.py --subset FD001

# 2. Train a model
python scripts/train.py --model lstm --subset FD001
python scripts/train.py --model gru --subset FD001
python scripts/train.py --model transformer --subset FD001
python scripts/train.py --model hybrid --subset FD001

# 3. Evaluate the trained model
python scripts/evaluate.py --model lstm --subset FD001

# 4. (Optional) Run exploratory data analysis
=======
# PHM-XAI: Engine-Aware Trustworthy Prognostics

End-to-end prognostics pipeline for **remaining useful life (RUL)** and **health index (HI)** estimation on the NASA C-MAPSS turbofan dataset. Point predictions are extended with calibrated uncertainty intervals, post-hoc explanations, and constraint-aware maintenance recommendations.

> Results are from project runs on C-MAPSS. This is a research implementation, not a production system. See [Implementation Status](#implementation-status) for what is complete and what is not.

**New here?** [ARCHITECTURE.md](ARCHITECTURE.md) documents the internal design — module responsibilities, artifact contracts, the conformal calibration math, the full decision policy, and current gaps.

---

## Contents

- [Overview](#overview)
- [Repository Layout](#repository-layout)
- [Setup](#setup)
- [Pipeline](#pipeline)
- [Decision Support](#decision-support)
- [HITL Review](#hitl-review)
- [Results Summary](#results-summary)
- [Implementation Status](#implementation-status)
- [Tests](#tests)

---

## Overview

The pipeline treats prognostics as a decision-support chain. A recommendation is considered trustworthy only when four questions are answered together:

1. **What is the predicted degradation state?** Joint RUL and HI estimation provides the point forecast.
2. **How uncertain is that forecast?** Engine-disjoint conformal calibration provides the uncertainty baseline; EARA-Conformal adds regime- and scale-aware interval sizing.
3. **Why did the model produce it?** Integrated Gradients, SHAP, temporal relevance, and ERI expose sensor-level evidence and flag explanation disagreement.
4. **What action is feasible?** A rule-based engine combines health state, uncertainty, ERI, and operational constraints into an auditable recommendation.

```text
NASA C-MAPSS → Preprocessing → Multi-task model → RUL / HI predictions
                                                         |
                                          ┌──────────────┴──────────────┐
                                   EARA-Conformal              IG, SHAP, ERI
                                   uncertainty                 explanations
                                          └──────────────┬──────────────┘
                                                  Decision engine
                                                  (rules + constraints)
                                                         |
                                               Auditable recommendation
                                                         |
                                              HITL review → feedback
                                              → policy refinement
```

Two invariants hold at every stage:

- **Engine-disjoint splits.** Windows from one engine are correlated, so train/validation and calibration/test splits are grouped by engine ID, and every stage asserts zero engine overlap before continuing.
- **Calibration never sees test targets.** Conformal quantiles are estimated on a calibration split carved out of the training set; the test set is used only to measure coverage.

---

## Repository Layout

```text
ARCHITECTURE.md  internal design, data contracts, known gaps
README.md        this file
configs/         subset configuration files (fd001–fd004.yaml) — see note below
data/
  external/      example decision input JSON files
  raw/           NASA C-MAPSS text files — not committed, add manually
  processed/     NumPy windows and labels — generated by preprocess.py
scripts/         CLI entry points for each pipeline stage
src/
  preprocessing/  loading, validation, labelling, selection, scaling, windowing
  models/         LSTM, GRU, Transformer, Hybrid, GRU-Att-Deg (+ losses, trainer, evaluator)
  uncertainty/    conformal prediction and EARA-Conformal
  explainability/ IG, SHAP, temporal relevance, ERI
  decision_engine/ rules, constraints, recommender
  hitl/           review workflow, feedback schema, policy refinement
  evaluation/     placeholder stubs — see Implementation Status
  utils/          logging (config/io/seed are stubs)
tests/            pytest suite
outputs/          generated artifacts — not committed
  checkpoints/    trained model weights, per model/subset
  figures/        EDA and experiment plots
  hitl/           feedback.jsonl — append-only expert feedback log
  logs/           pipeline logs
  models/         scalers
  reports/        HTML reports and CSV summaries
  results/        metrics, predictions, intervals, decisions
                  results/uncertainty/<variant>/<subset>/<model>.npz
  xai/            attribution arrays, ERI JSON, HTML reports
```

> **Config files are not wired up.** `configs/*.yaml` record the intended hyperparameters, but no script reads them — `src/utils/config.py::load_config` is a stub and each stage uses its own module-level constants. Treat the scripts as the source of truth until config loading lands. See [ARCHITECTURE.md](ARCHITECTURE.md#caveat-configuration-files-are-not-wired-up).

---

## Setup

Python 3.9+ required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins a CUDA 12.8 PyTorch build. For CPU-only or a different CUDA version, install a compatible PyTorch build first, then run `pip install -r requirements.txt`.

### Dataset

Place the raw C-MAPSS files in `data/raw/`:

```text
data/raw/
  train_FD001.txt   test_FD001.txt   RUL_FD001.txt
  train_FD002.txt   test_FD002.txt   RUL_FD002.txt
  train_FD003.txt   test_FD003.txt   RUL_FD003.txt
  train_FD004.txt   test_FD004.txt   RUL_FD004.txt
```

---

## Pipeline

Each stage depends on outputs from the previous one. `--subset all` and `--model all` iterate over every combination; a stage that fails for one combination is reported and skipped rather than aborting the run.

```powershell
# Exploratory data analysis (FD001 only, hard-coded)
>>>>>>> 232c8a9acfc44f8cf76034ecb550fec335210471
python main.py

# Preprocess (one subset or all)
python scripts/preprocess.py --subset FD001
python scripts/preprocess.py --subset all

# Train and evaluate
python scripts/train.py    --model gru --subset FD001
python scripts/evaluate.py --model gru --subset FD001

# Uncertainty intervals
python scripts/uncertainty.py --model gru --subset FD001 --n_regimes 7

# Explanations (one target or all)
python scripts/explain.py --model gru_att_deg --subset FD001 --target rul
python scripts/explain.py --model all --subset all --target all
```

<<<<<<< HEAD
### Output Locations

```
outputs/
├── checkpoints/{model}/best_model.pt     # Best model weights
├── results/{subset}_{model}_history.npz  # Training curves
├── results/{subset}_{model}_predictions.npz # Test predictions
├── reports/{subset}/                     # Validation summaries
├── figures/{subset}/                     # EDA plots
└── logs/                                 # Preprocessing, training, eval logs
```

---

## Configuration

Experiment parameters are defined in `configs/` (one file per C-MAPSS subset):

```yaml
dataset: FD001
window_size: 30
batch_size: 64
epochs: 50
learning_rate: 0.001
random_seed: 42
device: cuda
```

Default training hyperparameters (in `scripts/train.py`):

| Parameter | Default |
|-----------|---------|
| Window size | 30 |
| Batch size | 64 |
| Epochs | 50 |
| Learning rate | 0.001 |
| Hidden size | 128 |
| Layers | 2 |
| Dropout | 0.3 |
| RUL loss weight (α) | 1.0 |
| HI loss weight (β) | 0.5 |
| Validation split | 20% (engine-wise) |
| Early stopping patience | 10 |

---

## Evaluation Metrics

### RUL (Remaining Useful Life)

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error in cycles |
| **RMSE** | Root Mean Squared Error |
| **R²** | Coefficient of determination |
| **NASA Score** | Asymmetric penalty — late predictions penalized more heavily |

### Health Index (HI)

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error |
| **RMSE** | Root Mean Squared Error |
| **R²** | Coefficient of determination |

---

## Future Development

The following research objectives are planned as extensions to the completed prognostics core:

### Phase 2 — Uncertainty Quantification (O2)

- [ ] Conformal prediction intervals for RUL estimates
- [ ] Probability calibration for health index outputs
- [ ] Coverage and reliability metrics

### Phase 3 — Multi-Level Explainability (O3)

- [ ] SHAP-based sensor importance analysis
- [ ] Integrated Gradients attributions (Captum)
- [ ] Temporal attention visualization
- [ ] Explainable RUL Index (ERI)

### Phase 4 — Explainable Decision Intelligence (O4)

- [ ] Rule-based maintenance decision engine
- [ ] Operational constraint checking
- [ ] Action recommendation with traceable rationale

### Phase 5 — Human-in-the-Loop Learning (O5)

- [ ] Expert feedback capture and logging
- [ ] Model refinement from maintenance operator input

### Phase 6 — Multi-Dimensional Evaluation (O6)

- [ ] Benchmark predictive accuracy across models and C-MAPSS subsets
- [ ] Evaluate uncertainty reliability and calibration
- [ ] Evaluate explainability quality
- [ ] Evaluate maintenance decision effectiveness
- [ ] Conduct ablation studies and robustness testing
- [ ] Assess usability and user trust

---

## Contributing

Contributions are welcome. See **[CONTRIBUTING.md](CONTRIBUTING.md)** for setup instructions, coding standards, and the pull request workflow.

Areas where contributions are especially valuable:

- New prognostics model architectures
- Preprocessing improvements
- Evaluation and benchmarking
- Documentation and test coverage

---

## Citation

If you use PHM-XAI in your research, please cite:

```bibtex
@software{phm_xai,
  title  = {PHM-XAI: Explainable Prognostics and Health Monitoring},
  author = {Nagesh Tiwari and Suraj Vishwakarma and Shani Vishwakarma and Vishwa Save},
  year   = {2026}
}
```

---

<div align="center">
  <p>Built for predictive maintenance research on NASA C-MAPSS engine degradation data.</p>
</div>
=======
Valid model names: `lstm`, `gru`, `transformer`, `hybrid`, `gru_att_deg`  
Valid subsets: `FD001`, `FD002`, `FD003`, `FD004`  
Valid explain targets: `rul`, `hi`, `all`

### Uncertainty variants

`scripts/uncertainty.py` implements five calibration methods, selected with `--variant`:

| Variant | Interval | Character |
|---|---|---|
| `eara_conformal` *(default)* | `ŷ ± q̂_k · s(x)` | Per-regime, scale-adaptive |
| `adaptive_conformal` | `ŷ ± q̂ · s(x)` | Global, scale-adaptive |
| `conformal` | `ŷ ± q̂` | Global split conformal (baseline) |
| `cqr` | `ŷ ± (s(x) + q̂)` | Scale-based conformalized |
| `engine_joint_conformal` | `ŷ ± q̂` | One score per calibration engine |

`--n_regimes` (default 7) sets the k-means regime count used by `eara_conformal`. The default was selected by sweeping k = 2..10:

```powershell
python scripts/regime_sweep.py    # writes outputs/results/regime_sweep.json
```

Coverage is evaluated at nominal 80% / 90% / 95%, reporting empirical coverage, coverage error, MPIW, PINAW, per-regime coverage, and the stricter engine-level coverage. Results land in `outputs/results/uncertainty/<variant>/<subset>/`.

> The variants differ only in how the interval radius is calibrated; all five share the same engine-disjoint calibration split and the same `[0, 125]` clipping.

---

## Decision Support

The decision engine fuses RUL, HI, the conformal interval, and ERI into a risk score, classifies health and uncertainty, selects an action, then applies operational constraints. Constraints can only make an action more conservative, never less.

It normally consumes upstream artifacts directly — predictions, the 90% interval, and the ERI/top-sensor JSON from `scripts/explain.py` — and names every missing artifact if a stage has not been run.

```powershell
# Run against all subsets using default thresholds
python scripts/decision.py --subset all

# Pick a specific uncertainty variant and model
python scripts/decision.py --subset FD001 --model hybrid --uncertainty-variant adaptive_conformal

# Run with custom operational constraints
python scripts/decision.py `
  --subset FD001 `
  --minimum-safe-rul 10 `
  --maintenance-lead-time 10 `
  --no-maintenance-window
```

Outputs are written to `outputs/results/` (JSON) and `outputs/reports/` (HTML).

**Actions:** `CONTINUE_OPERATION`, `MONITOR`, `INSPECT`, `SCHEDULE_MAINTENANCE`, `URGENT_MAINTENANCE`. Each result includes health state, uncertainty level, explanation reliability, risk score, urgency index, risk components, reasons, constraint violations, and a `human_review` flag.

**Constraints** (all optional; unset means unconstrained):

| Flag | Effect |
|---|---|
| `--maintenance-window` / `--no-maintenance-window` | Whether scheduled maintenance can be performed now |
| `--minimum-safe-rul <cycles>` | Minimum RUL to defer maintenance |
| `--maintenance-lead-time <cycles>` | Cycles required before maintenance can begin |
| `--max-allowable-delay <cycles>` | Longest wait the operator will accept |
| `--resource-availability <true\|false>` | Whether crew and parts are available |

For testing the engine in isolation, `data/external/decision_input*.json` contains example records (healthy, degrading, critical). A record needs `rul`, `hi`, `rul_lower`, `rul_upper`, and `eri`; `top_k_sensors` and engine identifiers are optional.

---

## HITL Review

The HITL runner consumes the JSON produced by `scripts/decision.py`, creates review items for decisions flagged `human_review`, and writes JSON results, an HTML report, and a log. It then applies conservative policy refinement from accumulated expert feedback.

```powershell
python scripts/hitl.py --subset FD001 --model gru
python scripts/hitl.py --subset all --model all

# Resolve a queued review
python scripts/hitl.py `
  --input outputs/results/decision_FD001_gru.json `
  --review-id decision-0-0 `
  --reviewer operator-1 `
  --action APPROVE `
  --confidence 0.9 `
  --reason "Reviewed by maintenance control"
```

Supported actions: `APPROVE`, and `OVERRIDE` (requires `--override-action`, which must differ from the AI action). Resolving a review requires `--review-id`, `--reviewer`, `--action`, and `--confidence` together.

Feedback is appended to the append-only log `outputs/hitl/feedback.jsonl`. An alternative action is adopted only with repeated, high-confidence, high-consensus expert support (≥3 observations, weighted support ≥2.0, consensus ≥0.70), and `URGENT_MAINTENANCE` is never automatically weakened. Constraints are re-checked after every adaptation.

> **HITL does not retrain the model.** It refines the decision *policy* only — `prognostic_model_updated` is `false` on every output. A review submitted in a run is recorded but excluded from that same run's adaptation pass, so it becomes evidence for the next run rather than leaking into the decision it just reviewed.

---

## Results Summary

Results below are from specific project runs and depend on the preprocessing config, checkpoint, and seed. Re-run the relevant scripts before using them as a formal comparison.

| Area | Key result |
|---|---|
| Point prediction (FD001, GRU-Att-Deg) | RUL MAE 10.05, RMSE 13.64, R² 0.884, NASA Score 308.63 |
| Point prediction (FD004, Transformer) | RUL MAE 13.56, R² 0.808 |
| Uncertainty (GRU, split conformal, 90%) | Coverage 0.880 / 0.897 / 0.850 / 0.928 on FD001–FD004; MPIW 42.06 / 57.88 / 37.87 / 70.81 cycles |
| Explanation reliability (mean ERI) | 0.880 (HI), 0.872 (RUL) across subsets and models |

ERI is the Explainability Reliability Index — cross-method agreement between Integrated Gradients, SHAP, and temporal relevance, with an uncorrelated pair scoring 0.5 (see [ARCHITECTURE.md](ARCHITECTURE.md#eri--explainability-reliability-index-eripy)).

EARA-Conformal adaptive figures are not reported here pending repeated engine-level evaluation. The adaptive variants size intervals from a learned per-sample scale `s(x)`; that head is now supervised during training, so **these figures should be regenerated from retrained checkpoints** — see below.

---

## Implementation Status

| Component | Status |
|---|---|
| Preprocessing and windowing | Complete |
| Joint RUL/HI prediction (5 model families) | Complete |
| Conformal uncertainty — all five variants | Implemented; multi-seed validation pending |
| Explainability (IG, SHAP, temporal relevance, ERI) | Implemented; faithfulness validation pending |
| Decision engine (rules, constraints, HTML reports) | Complete |
| HITL review and policy refinement | Complete; model retraining intentionally out of scope |
| `src/evaluation/` (benchmark, ablation, robustness) | Placeholder stubs |
| Config file loading | Not implemented — scripts use module-level constants |

Additional gaps that affect how results should be read:

- **Existing checkpoints have an untrained scale head.** `MultiTaskLoss` now includes a Gaussian NLL term that supervises `s(x)` toward `|y − ŷ|`, resolving the earlier gap where no loss referenced the scale head at all. Checkpoints trained before this change still carry an untrained head, so **retrain before trusting adaptive interval widths**. See [ARCHITECTURE.md](ARCHITECTURE.md#known-gaps).
- **`src/evaluation/` returns hard-coded values.** The real metrics live in `src/models/evaluator.py`; ablation and robustness experiments are not implemented.
- **`cqr` is not conventional CQR** — the checkpoint exposes no trained quantile heads, so it uses `prediction ± scale` as conditional bounds.

[ARCHITECTURE.md](ARCHITECTURE.md#known-gaps) documents all of these with the code locations involved.

---

## Tests

```powershell
python -m pytest -q
```

The decision and uncertainty suites are the meaningful regression net: they cover the full O4 rule/constraint policy and the O5 HITL workflow, and run without the dataset.

| File | Scope | Needs data |
|---|---|---|
| `tests/test_decision.py` | Decision rules, constraints, risk scoring, HITL review and adaptation | No |
| `tests/test_uncertainty.py` | Conformal quantiles, interval clipping, coverage semantics | No |
| `tests/test_preprocessing.py` | Short test engines are padded, not dropped | **Yes — `data/raw/`** |
| `tests/test_models.py`, `tests/test_xai.py` | Placeholder | No |
>>>>>>> 232c8a9acfc44f8cf76034ecb550fec335210471
