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
python main.py
```

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

## Testing

Run the full test suite:

```bash
pytest tests/ -v
```

Run specific modules:

```bash
pytest tests/test_preprocessing.py -v
pytest tests/test_models.py -v
```

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

- [ ] Benchmarking framework across models and subsets
- [ ] Ablation studies and robustness testing

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
